#include <Adafruit_TinyUSB.h>
#include <bluefruit.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BME680.h"
#include <as3933.h>
#include "Arduino.h"
#include "epd2in13_V3.h"
#include "epdpaint.h"
#include "imagedata.h"
#include "serial_frame_util.h"


#define CMD_VLC_FRAME_DATA_REQ_SER   "AA"// request sensor data from node using VLC


//EXAMPLE GATEWAY SENDS A FRAME: #BLF,FC,E4,82,DC,DD,6D,0D,0A
//-----
#define COLORED 0
#define UNCOLORED 1
#define BME_CS 12
#define AS3833_CS 17
#define SEALEVELPRESSURE_HPA (1013.25)

unsigned char image[4000];  //1050];
Epd epd;
Adafruit_BME680 bme(BME_CS);  // hardware SPI
As3933 asTag(SPI, AS3833_CS);

//******************THROUGPUT**********************
// https://github.com/adafruit/Adafruit_nRF52_Arduino/blob/master/libraries/Bluefruit52Lib/examples/Peripheral/throughput/throughput.ino

// // data to send in the throughput test
// char test_data[256] = { 0 };
// // Number of packet to sent
// // actualy number of bytes depends on the MTU of the connection
// #define PACKET_NUM    1000//5675//4207//2742//1341//1000// 1952//1574 //1000//10167 //1574
// BLEUart bleuart;

// uint32_t rxCount     = 0;
// uint32_t rxStartTime = 0;
// uint32_t rxLastTime  = 0;

// bool test_running     = false; // Flag to control test execution
// bool test_completed   = false; // Flag to indicate test completion
// uint32_t lastTestTime = 0; // Store the time of the last test
// const uint32_t testInterval = 5000; // 10 seconds
// *******************SENSOR TX *******************************
BLEUart bleuart;
// *************************************************************

// These define's must be placed at the beginning before #include "NRF52TimerInterrupt.h"
#define TIMER_INTERRUPT_DEBUG 0
#define _TIMERINTERRUPT_LOGLEVEL_ 3
#define _PWM_LOGLEVEL_ 0

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include "NRF52TimerInterrupt.h"

// Select false to use PWM
#define USING_TIMER false
#include "nRF52_PWM.h"
#define TIMER0_INTERVAL_MS 500
#define IN1 15       //NBVLC RX pin
#define pinToUse 16  //NBVLC TX pin

/**********Add**********/
#define FRAME_BUFFER_SIZE 64
uint8_t frameBuffer[FRAME_BUFFER_SIZE]; // Buffer to store received frame
volatile bool frameReceived = false;           // Flag indicating a complete frame received
// Define start and end frame markers
const uint8_t START_FRAME = 0xAA; // Start of Text (STX)
const uint8_t END_FRAME = 0xEE;   // End of Text (ETX)
// Addresses and commands for protocol
const uint8_t ADDRESS_Source = 0x01; // Source address (Node). Range is 00-F0
const uint8_t ADDRESS_Destination = 0xF1; // Destination address (Gateway). Range is F1-FF

const uint8_t CMD_SEND = 0x01;   // Example command to send data
const uint8_t CMD_RECEIVE = 0x02; // Example command to receive data


const uint8_t CMD_Running_State = 0x03; //Power mode
const uint8_t CMD_Wakeup = 0x04; //Wakeup 
const uint8_t CMD_Communcation_mode = 0x05; //Communcation mode, choose vlc or ble
const uint8_t CMD_VLC_Protocol = 0x06; //vlc bit, byte, compact level
const uint8_t CMD_VLC_Intervel = 0x07; //Intervel
const uint8_t CMD_PWM = 0x08; //PWM
const uint8_t CMD_VLC_Unit = 0x09; //MTU
const uint8_t CMD_BLE_Protocol = 0x0A; //BLE_Protocol
const uint8_t CMD_BLE_Intervel = 0x0B; //BLE_Intervel
const uint8_t CMD_PHY_Rate = 0x0C; //PHY_Rate
const uint8_t CMD_MTU = 0x0D; //MTU
const uint8_t CMD_BLE_Unit = 0x0E; //Unit
const uint8_t CMD_BLE_OFF_ON = 0x0F; //BLE OFF/ON
const uint8_t CMD_BLE_FAST_TIME = 0X11;

const uint8_t CMD_VLC_OFF_ON = 0x0A; //VLC OFF/ON
const uint8_t CMD_EINK_OFF_ON = 0x0B; //VLC OFF/ON
const uint8_t CMD_SENSING_OFF_ON = 0x0C; //VLC OFF/ON

const uint8_t CMD_VLC_SEND_Auto = 0x10; //SEND VLC AUTO FOR TEST
const uint8_t CMD_BLE_SEND_Auto = 0x20; //SEND BLE AUTO FOR TEST

volatile uint8_t frameQueue[FRAME_BUFFER_SIZE];
volatile size_t frameQueueLen = 0;
volatile size_t frameSendIndex = 0;
volatile bool frameSending = false;



uint8_t BLE_ADDRESS[6] = {0xD6, 0xD6, 0x70, 0x4D, 0x29, 0x6B}; // ble address (Node). replace it with actual mac
// uint8_t BLE_ADDRESS[6] = {0xFC, 0xE4, 0x82, 0xDC, 0xDD, 0x6D}; // ble address (Node). replace it with actual mac

size_t frameIndex = 0;              // Index for the current frame byte
size_t frameIndex_clone = 0;              // Index for the current frame byte
volatile bool frameFlag = false;    // Flag indicating a complete frame received

volatile bool ble_frameFlag = false;    // Flag indicating a complete ble frame received

int receivedData = 0;  // The byte currently being received
volatile uint8_t bitIndex = 0;      // Tracks the bit position in the byte
volatile bool dataReady = false;    // Flag indicating a byte is fully received
uint32_t t0 = 0, dt = 0;    // Timing variables

//********** BLE **********//
#define BLE_FRAME_BUFFER_SIZE 64
uint8_t BLE_frameBuffer[BLE_FRAME_BUFFER_SIZE];  // Circular buffer for incoming data
size_t BLE_bufferHead = 0;  // Points to where new data is added
size_t BLE_bufferTail = 0;  // Points to where processing starts
size_t BLE_frameSize = 0;
/**********Add**********/

//creates pwm instance
nRF52_PWM* PWM_Instance;

float frequency = 38000.0f;
float dutyCycle = 0.0f;
uint32_t time_t_now = 0;
uint32_t time_t_old = 0;
uint32_t _c = 0x00ff107f;
int _start_send = 0;
int _bit_i = 0;
volatile uint64_t Timer0Count = 0;
int t1 = Timer0Count;
int state = 0;
int long ls[64], ts[64];
int i = 0;
int long len = 0;

// Init NRF52 timer NRF_TIMER1
NRF52Timer ITimer0(NRF_TIMER_4);
float NNF = 1.28;  //Scale factor to tune NEC bit time

BLEService myService = BLEService("19b10010e8f2537e4f6cd104768a1214");
BLECharacteristic myChar = BLECharacteristic("19b10020e8f2537e4f6cd104768a1214");

/**********Add**********/
// UUIDs for custom service and characteristics
#define CUSTOM_SERVICE_UUID       "19b10010e8f2537e4f6cd104768a1214"
#define CHAR_RX_UUID              "19b10020e8f2537e4f6cd104768a12ee" // For receiving frames
#define CHAR_TX_UUID              "19b10020e8f2537e4f6cd104768a12aa" // For sending frames
// Create the custom BLE service and characteristics
BLEService customService(CUSTOM_SERVICE_UUID);
BLECharacteristic charRX(CHAR_RX_UUID);
BLECharacteristic charTX(CHAR_TX_UUID);
/**********Add**********/


//*********************************** ADDED FOR TX and RX RX FRAME ***************************************************
// Frame markers and command types
#define START_MARKER    0xAA        // Frame start marker (binary: 10101010)
#define END_MARKER      0x55        // Frame end marker (binary: 01010101)
#define MAX_PAYLOAD_SIZE 16         // Maximum payload size in bytes

// Command types for different data types
#define CMD_SENSOR      0x01        // Sending only sensor data
#define CMD_LOCATION    0x02        // Sending only location data
#define CMD_COMBINED    0x03        // Sending both sensor and location data
#define CMD_ACK         0x04        // Acknowledgment message
#define CMD_NACK        0x05        // Negative acknowledgment
#define CMD_REQ_SENSOR  0x06        // Request sensor data
#define CMD_REQ_LOC     0x07        // Request location data
#define CMD_REQ_BOTH    0x08        // Request both sensor and location data
#define MY_ADDRESS      0x02       // Add this with other #define statements



// Frame size calculations
#define FRAME_SIZE_BYTES  23        // Total frame size: header(4) + payloadInfo(17) + footer(2)
#define FRAME_SIZE_BITS   (FRAME_SIZE_BYTES * 8)  // 184 bits total
#define LAST_CHUNK_BITS   ((FRAME_SIZE_BITS % 32) ? (FRAME_SIZE_BITS % 32) : 32)  // 24 bits in last chunk
#define FRAME_CHUNKS    6           // Number of 32-bit chunks needed: ceil(184/32) = 6

// Transmission timing constants
#define CHUNK_DELAY_MS    100        // Delay between chunk transmissions
#define FRAME_DELAY_MS    5000      // Delay between complete frames

// Frame reception buffer
uint32_t receivedDataFrame[FRAME_CHUNKS];  // Buffer for received chunks
int receivedChunks = 0;               // Count of received chunks
bool isReceiving = false;             // Frame reception in progress flag



// Define constants for your parameters if helpful
#define PHY_1M 1
#define PHY_2M 2
#define PHY_3M 3 // For Long Range

// <<< ADDED >>> Global variables for connection state and parameters
uint16_t _conn_handle = BLE_CONN_HANDLE_INVALID; // Store the connection handle
// Store the *last requested* values to avoid redundant requests
// Initialize with values that likely won't match the first request
int last_requested_phy_pref = 0;       // 0: Unset, 1: 1M, 2: 2M, 3: Coded
uint16_t last_requested_interval_units = 0; // In 1.25ms units
uint16_t last_requested_mtu = 0;       // MTU size (default is usually 23)

// Define a struct to hold the return values
typedef struct {
    uint8_t addr_ble_rec[6];
    uint8_t command;
    uint8_t length;
    size_t index;
} BLE_Data;
BLE_Data processFramePayload(size_t start, size_t frameSize);


#define MAX_TOKENS  16   // adjust up if you need more data bytes

#define MAX_FRAME_SIZE_serial 128
uint8_t frameBuf_serial[MAX_FRAME_SIZE_serial];
size_t  frameHead_serial     = 0;
bool    frameReceived_serial = false;

/*
 * Optimized Sensor Data Structure (10 bytes)
 * All values are stored in fixed-point format to save space
 */
struct __attribute__((packed)) SensorData {
    int16_t temperature;    // Temperature in °C * 10 (-400 to 850 for -40.0 to 85.0°C)
    uint16_t pressure;      // Pressure in hPa * 10 - 300 (3000 to 11000 for 300.0 to 1100.0 hPa)
    uint16_t humidity;      // Humidity in % * 10 (0 to 1000 for 0.0 to 100.0%)
    uint16_t gas;          // Gas resistance in kΩ * 100 (0 to 20000 for 0.0 to 200.0 kΩ)
};

/*
 * Optimized Location Data Structure (6 bytes)
 * All coordinates are stored with 2 decimal places
 */
struct __attribute__((packed)) LocationData {
    int16_t x;             // X coordinate * 100 (-1000 to 1000 for -10.00 to 10.00m)
    int16_t y;             // Y coordinate * 100 (-1000 to 1000 for -10.00 to 10.00m)
    int16_t z;             // Z coordinate * 100 (-1000 to 1000 for -10.00 to 10.00m)
};

/*
 * Frame Structure (23 bytes total)
 * Optimized for efficient transmission while maintaining data integrity
 */
struct __attribute__((packed)) Frame {
    uint8_t startMarker;    // Frame start marker (0xAA)
    uint8_t sourceAddr;     // Source device address
    uint8_t destAddr;       // Destination device address
    uint8_t command;        // Command type (sensor/location/combined)
    uint8_t payloadSize;    // Size of payload in bytes
    uint8_t payload[MAX_PAYLOAD_SIZE];  // Payload data (sensor and/or location)
    uint8_t checksum;       // Simple checksum for error detection
    uint8_t endMarker;      // Frame end marker (0x55)
};

// Frame chunk layout in 32-bit chunks
/*
 * Chunk 0: [startMarker][sourceAddr][destAddr][command]     - Frame header
 * Chunk 1: [payloadSize][payload[0:2]]                     - Start of payload
 * Chunk 2: [payload[3:6]]                                  - Payload data
 * Chunk 3: [payload[7:10]]                                 - Payload data
 * Chunk 4: [payload[11:14]]                                - Payload data
 * Chunk 5: [payload[15]][checksum][endMarker][padding]     - End of payload and frame footer
 */


/*  add to collect diverse factors */
CollectedData_t collected_data;
void bme_setup();
void bme_print();
void as_setup();
void nbvlc_on();
void nbvlc_off();
void epd_message();


// Function to calculate checksum for received frame
uint8_t calculateChecksum(const Frame* frame) {
    uint8_t sum = 0;
    sum += frame->sourceAddr;
    sum += frame->destAddr;
    sum += frame->command;
    sum += frame->payloadSize;
    
    for(int i = 0; i < frame->payloadSize; i++) {
        sum += frame->payload[i];
    }
    return sum;
}

// Function to validate received frame
bool validateFrame(const Frame* frame) {
    // Check frame markers
    if(frame->startMarker != START_MARKER || frame->endMarker != END_MARKER) {
        Serial.println("Error: Invalid frame markers");
        return false;
    }

    // Check if frame is intended for this device
    if(frame->destAddr != MY_ADDRESS && frame->destAddr != 0xFF) {  // 0xFF for broadcast
        Serial.println("Error: Frame not intended for this device");
        return false;
    }

    // Check payload size
    if(frame->payloadSize > MAX_PAYLOAD_SIZE) {
        Serial.println("Error: Payload size exceeds maximum");
        return false;
    }

    // Validate command type
    if(frame->command != CMD_SENSOR && 
       frame->command != CMD_LOCATION && 
       frame->command != CMD_COMBINED && 
       frame->command != CMD_ACK && 
       frame->command != CMD_NACK && 
       frame->command != CMD_REQ_SENSOR && 
       frame->command != CMD_REQ_LOC && 
       frame->command != CMD_REQ_BOTH) {
        Serial.println("Error: Invalid command type");
        return false;
    }

    // Verify payload size matches command type
    switch(frame->command) {
        case CMD_SENSOR:
            if(frame->payloadSize != sizeof(SensorData)) {
                Serial.println("Error: Invalid sensor data size");
                return false;
            }
            break;
        case CMD_LOCATION:
            if(frame->payloadSize != sizeof(LocationData)) {
                Serial.println("Error: Invalid location data size");
                return false;
            }
            break;
        case CMD_COMBINED:
            if(frame->payloadSize != (sizeof(SensorData) + sizeof(LocationData))) {
                Serial.println("Error: Invalid combined data size");
                return false;
            }
            break;
        case CMD_REQ_SENSOR:
        case CMD_REQ_LOC:
        case CMD_REQ_BOTH:
        case CMD_ACK:
        case CMD_NACK:
            if(frame->payloadSize != 0) {
                Serial.println("Error: Request/ACK/NACK should have no payload");
                return false;
            }
            break;
    }

    // Verify checksum
    uint8_t calculatedChecksum = calculateChecksum(frame);
    if(calculatedChecksum != frame->checksum) {
        Serial.println("Error: Checksum mismatch");
        return false;
    }

    return true;
}


/***********Add functions**********/
void processReceivedData(uint8_t cmd, uint8_t* data1) {
  // Process based on command type
  switch(cmd) {
      case CMD_REQ_SENSOR: {
          Serial.println("Received sensor data request");
          
          // Create response frame with sensor data
          Frame responseFrame;
          responseFrame.startMarker = START_MARKER;
          responseFrame.sourceAddr = MY_ADDRESS;
          responseFrame.destAddr = data1[0];
          responseFrame.command = CMD_SENSOR;
          
          // Read current sensor data
          // float temperature = 68.9;
          // float pressure = 500.3;
          // float humidity = 98.5;
          // float gas = 95.75;
          
          bme_setup();
          bme_print();
          float temperature = bme.temperature;
          float pressure = bme.pressure / 100.0;
          float humidity = bme.humidity;
          float gas = bme.gas_resistance / 1000.0;


          SensorData sensorData = prepareSensorData(temperature, pressure, humidity, gas);
          
          responseFrame.payloadSize = sizeof(SensorData);
          memcpy(responseFrame.payload, &sensorData, sizeof(SensorData));
          responseFrame.checksum = calculateChecksum(&responseFrame);
          responseFrame.endMarker = END_MARKER;
          
          delay(1000);  // Small delay before responding
          transmitFrame(responseFrame);
          Serial.print("T:"); Serial.print(temperature, 1);
          Serial.print("°C P:"); Serial.print(pressure, 1);
          Serial.print("hPa H:"); Serial.print(humidity, 1);
          Serial.print("% G:"); Serial.print(gas, 2);
          Serial.println("kΩ");
          break;
      }
      
      case CMD_REQ_LOC: {
          Serial.println("Received location data request");
          
          // Create response frame with location data
          Frame responseFrame;
          responseFrame.startMarker = START_MARKER;
          responseFrame.sourceAddr = MY_ADDRESS;
          responseFrame.destAddr = data1[0];
          responseFrame.command = CMD_LOCATION;
          
          // Define current location coordinates
          float x_coord = 9.23;  // X coordinate in meters
          float y_coord = -4.56; // Y coordinate in meters
          float z_coord = 9.89;  // Z coordinate in meters

          LocationData locationData = prepareLocationData(x_coord, y_coord, z_coord);
          
          responseFrame.payloadSize = sizeof(LocationData);
          memcpy(responseFrame.payload, &locationData, sizeof(LocationData));
          responseFrame.checksum = calculateChecksum(&responseFrame);
          responseFrame.endMarker = END_MARKER;
          
          delay(1000);  // Small delay before responding
          transmitFrame(responseFrame);
          Serial.print("X:");Serial.print(x_coord,2);
          Serial.print("m Y:");Serial.print(y_coord,2);
          Serial.print("m Z:");Serial.print(z_coord,2);
          Serial.println("m");

          break;
      }
      
      case CMD_REQ_BOTH: {
          Serial.println("Received combined data request");
          
          // Create response frame with both sensor and location data
          Frame responseFrame;
          responseFrame.startMarker = START_MARKER;
          responseFrame.sourceAddr = MY_ADDRESS;
          responseFrame.destAddr = data1[0];
          responseFrame.command = CMD_COMBINED;
          
          // Get current sensor and location data
          // float temperature = 68.9;
          // float pressure = 500.3;
          // float humidity = 98.5;
          // float gas = 95.75;
          
          bme_setup();
          bme_print();
          float temperature = bme.temperature;
          float pressure    = bme.pressure / 100.0;
          float humidity    = bme.humidity;
          float gas         = bme.gas_resistance / 1000.0;


          SensorData sensorData = prepareSensorData(temperature, pressure, humidity, gas);

          // Define current location coordinates
          float x_coord = 9.23;  // X coordinate in meters
          float y_coord = -4.56; // Y coordinate in meters
          float z_coord = 9.89;  // Z coordinate in meters

          LocationData locationData = prepareLocationData(x_coord, y_coord, z_coord);
          
          responseFrame.payloadSize = sizeof(SensorData) + sizeof(LocationData);
          memcpy(responseFrame.payload, &sensorData, sizeof(SensorData));
          memcpy(responseFrame.payload + sizeof(SensorData), &locationData, sizeof(LocationData));
          responseFrame.checksum = calculateChecksum(&responseFrame);
          responseFrame.endMarker = END_MARKER;
          
          delay(1000);  // Small delay before responding
          transmitFrame(responseFrame);
          Serial.print("T:"); Serial.print(temperature, 1);
          Serial.print("°C P:"); Serial.print(pressure, 1);
          Serial.print("hPa H:"); Serial.print(humidity, 1);
          Serial.print("% G:"); Serial.print(gas, 2);
          Serial.print("kΩ X:");Serial.print(x_coord,2);
          Serial.print("m Y:");Serial.print(y_coord,2);
          Serial.print("m Z:");Serial.print(z_coord,2);
          Serial.println("m");


          break;
      }
      case CMD_ACK:
          Serial.println("Received data request");
          break;
  }

}
/********end function***************/



// Optimized sensor data preparation (reduced precision for pressure to fit in 16 bits)
SensorData prepareSensorData(float temp, float press, float hum, float gas) {
    SensorData data;
    data.temperature = (int16_t)(temp * 10.0);                      // -40.0 to 85.0°C
    data.pressure    = (uint16_t)((press - 300.0) * 10.0);          // 300.0 to 1100.0 hPa, stored as offset from 300
    data.humidity    = (uint16_t)(hum * 10.0);                      // 0.0 to 100.0%
    data.gas         = (uint16_t)(min(gas * 100.0, 20000.0));       // 0.0 to 200.0 kΩ, with 0.01 kΩ precision
    return data;
}

LocationData prepareLocationData(float x, float y, float z) {
    LocationData data;
    x = constrain(x, -10.0, 10.0);
    y = constrain(y, -10.0, 10.0);
    z = constrain(z, -10.0, 10.0);
    
    data.x = (int16_t)(x * 100.0);
    data.y = (int16_t)(y * 100.0);
    data.z = (int16_t)(z * 100.0);
    return data;
}

// Function to send 32 bits of frame data
void send_32bits(uint32_t chunk)
{
    _c = chunk;  // Store chunk in global variable for state machine
    _start_send = 1;  // Start transmission state machine
    while(_start_send != 0) {  // Wait for transmission to complete
        delay(1);
    }
}

/*
 * Debug function to print chunk contents
 * Displays the contents of each chunk in a human-readable format
 * based on the command type (sensor/location/combined)
 */
void debugPrintChunk(int chunkNum, uint32_t chunk, uint8_t command) {
    Serial.print("Chunk "); Serial.print(chunkNum); 
    Serial.print(" (0x"); Serial.print(chunk, HEX); Serial.print("): ");
    
    switch(chunkNum) {
        case 0:
            Serial.print("[Start:0x"); Serial.print(chunk & 0xFF, HEX);
            Serial.print("][Src:0x"); Serial.print((chunk >> 8) & 0xFF, HEX);
            Serial.print("][Dest:0x"); Serial.print((chunk >> 16) & 0xFF, HEX);
            Serial.print("][Cmd:0x"); Serial.print((chunk >> 24) & 0xFF, HEX);
            Serial.println("]");
            break;
            
        case 1:
            Serial.print("[Size:0x"); Serial.print(chunk & 0xFF, HEX); Serial.print("]");
            if(command == CMD_SENSOR) {
                uint16_t temp = (chunk >> 8) & 0xFFFF;
                //Serial.print("[Temp:"); Serial.print((int16_t)temp/10.0f); Serial.print("°C]");
            }
            Serial.println();
            break;
            
        case 2:
            if(command == CMD_SENSOR) {
                uint16_t press = chunk & 0xFFFF;
                uint16_t hum = (chunk >> 16) & 0xFFFF;
                //Serial.print("[Press:"); Serial.print((press/10.0f) + 300.0f); Serial.print("hPa]");
                //Serial.print("[Hum:"); Serial.print(hum/10.0f); Serial.println("%]");
            } else if(command == CMD_LOCATION) {
                int16_t x = chunk & 0xFFFF;
                int16_t y = (chunk >> 16) & 0xFFFF;
                //Serial.print("[X:"); Serial.print(x/100.0f); Serial.print("m]");
                //Serial.print("[Y:"); Serial.print(y/100.0f); Serial.println("m]");
            }
            break;
            
        case 3:
            if(command == CMD_SENSOR) {
                uint16_t gas = chunk & 0xFFFF;
                //Serial.print("[Gas:"); Serial.print(gas/100.0f); Serial.println("kΩ]");
            } else if(command == CMD_LOCATION) {
                int16_t z = chunk & 0xFFFF;
                //Serial.print("[Z:"); Serial.print(z/100.0f); Serial.println("m]");
            } else if(command == CMD_COMBINED) {
                uint16_t temp = chunk & 0xFFFF;
                uint16_t press = (chunk >> 16) & 0xFFFF;
                //Serial.print("[Temp:"); Serial.print((int16_t)temp/10.0f); Serial.print("°C]");
                //Serial.print("[Press:"); Serial.print((press/10.0f) + 300.0f); Serial.println("hPa]");
            }
            break;
            
        case 4:
            if(command == CMD_COMBINED) {
                uint16_t hum = chunk & 0xFFFF;
                uint16_t gas = (chunk >> 16) & 0xFFFF;
                //Serial.print("[Hum:"); Serial.print(hum/10.0f); Serial.print("%]");
                //Serial.print("[Gas:"); Serial.print(gas/100.0f); Serial.println("kΩ]");
            }
            break;
            
        case 5:
            if(command == CMD_COMBINED) {
                int16_t x = chunk & 0xFFFF;
                int16_t y = (chunk >> 16) & 0xFFFF;
                //Serial.print("[X:"); Serial.print(x/100.0f); Serial.print("m]");
                //Serial.print("[Y:"); Serial.print(y/100.0f); Serial.print("m]");
            }
            // For all commands, print the frame footer components
            Serial.print("[Payload[15]:0x"); Serial.print(chunk & 0xFF, HEX);
            Serial.print("][Checksum:0x"); Serial.print((chunk >> 8) & 0xFF, HEX);
            Serial.print("][EndMarker:0x"); Serial.print((chunk >> 16) & 0xFF, HEX);
            Serial.print("][Padding:0x"); Serial.print((chunk >> 24) & 0xFF, HEX);
            Serial.println("]");
            break;
    }
}


/*
 * Function to transmit a complete frame
 * 1. Converts frame structure to array of 32-bit chunks
 * 2. Sends each chunk with debug output
 * 3. Adds delay between chunks for reliable transmission
 */
void transmitFrame(const Frame& frame) {
    const uint32_t* chunks = (const uint32_t*)&frame;
    
    // Serial.println("\n--- Transmitting Frame ---");
    //set 1 bit, reprsent sending
    collected_data.running_mode |= (1 << 1); 
    collected_data.vlc_communication_volume = 4*FRAME_CHUNKS;
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(20);
    // Send all 6 chunks with consistent timing
    for(int i = 0; i < FRAME_CHUNKS; i++) {
        // debugPrintChunk(i, chunks[i], frame.command);
        nbvlc_on();
        send_32bits(chunks[i]);
        nbvlc_off();
        delay(CHUNK_DELAY_MS);
    }
    delay(200);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(10);
    // CLear bit 1 
    collected_data.running_mode &= ~(1 << 1);
    collected_data.vlc_communication_volume = 0;
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(10);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(10);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(10);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    delay(10);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    // nbvlc_on();
    // Serial.println("--- Frame Complete ---\n");
}


// ************************************************************************************************************
BLEDis bledis;  // DIS (Device Information Service) helper class instance
BLEBas blebas;  // BAS (Battery Service) helper class instance

uint8_t char_m = 0;  //used to switch on test modes = value updated by myChar
uint8_t tm = 0;      //current test mode

void myChar_write_cb(uint16_t len, BLECharacteristic* chr, uint8_t* data, uint16_t offset) {
  char_m = *data;
}

#define TIMER_INTERVAL_US       (TIMER0_INTERVAL_MS * 100)  // 500ms / 10 = 50us
#define TIMER_INTERVAL_MS       (TIMER_INTERVAL_US / 1000.0)  // 0.05ms
#define DESIRED_DELAY_MS        300
// #define DELAY_BETWEEN_BYTES     ((int)(DESIRED_DELAY_MS / TIMER_INTERVAL_MS))  // = 6000
#define DELAY_BETWEEN_BYTES     6000
// void TimerHandler0() {
//     Timer0Count++;
//     static bool toggle0 = !toggle0;

//     if (!_start_send) return;

//     switch (_start_send) {
//         case 1:  updateDC(50); time_t_now = 0; _start_send = 2; break;
//         case 2:  if (++time_t_now >= 131 * NNF) _start_send = 3; break;
//         case 3:  updateDC(0); time_t_now = 0; _start_send = 4; break;
//         case 4:  if (++time_t_now >= 65 * NNF) { _start_send = 5; _bit_i = 0; } break;
//         case 5:  updateDC(50); time_t_now = 0; _start_send = 6; break;
//         case 6:  if (++time_t_now >= 7 * NNF) _start_send = 7; break;
//         case 7:  updateDC(0); time_t_now = 0; _start_send = ((0x00000001 << _bit_i) & _c) ? 8 : 9; break;
//         case 8:  if (++time_t_now >= 21 * NNF) _start_send = 10; break;
//         case 9:  if (++time_t_now >= 7 * NNF) _start_send = 10; break;
//         case 10: _start_send = (++_bit_i >= 32) ? 11 : 5; break;
//         case 11: updateDC(50); time_t_now = 0; _start_send = 12; break;
//         case 12: if (++time_t_now >= 7 * NNF) _start_send = 13; break;
//         // case 13: updateDC(0); _start_send = 0; break;
//         case 13:
//             updateDC(0);
//             time_t_now = 0;
//             _start_send = 14; // Delay before next byte
//             break;

//         case 14:
//             if (++time_t_now >= DELAY_BETWEEN_BYTES) { // Adjust as needed
//                 frameSendIndex++;
//                 if (frameSendIndex < frameQueueLen) {
//                   encodeAndStartSend(frameQueue[frameSendIndex]);  
//                 } else {
//                     _start_send = 0;
//                     frameSending = false;
//                 }
//             }
//             break;
//     }
// }

void TimerHandler0() {
  static bool toggle0 = false;

  // Flag for checking to be sure ISR is working as Serial.print is not OK here in ISR
  Timer0Count++;

  //timer interrupt toggles pin LED_BUILTIN
  //digitalWrite(LED_BUILTIN, toggle0);
  toggle0 = !toggle0;

  switch (_start_send) {
    case 1:
      {
        updateDC(50);
        time_t_now = 0;
        _start_send = 2;
        break;
      }
    case 2:
      {
        time_t_now++;
        if (time_t_now >= 131 * NNF) _start_send = 3;
        break;
      }
    case 3:
      {
        updateDC(0);
        time_t_now = 0;
        _start_send = 4;
        break;
      }
    case 4:
      {
        time_t_now++;
        if (time_t_now >= 65 * NNF) {
          _start_send = 5;
          _bit_i = 0;
        }
        break;
      }
    case 5:
      {
        updateDC(50);
        time_t_now = 0;
        _start_send = 6;
        break;
      }
    case 6:
      {
        time_t_now++;
        if (time_t_now >= 7 * NNF) _start_send = 7;
        break;
      }
    case 7:
      {
        updateDC(0);
        time_t_now = 0;
        if (((0x00000001 << _bit_i) & (_c))) _start_send = 8;
        else _start_send = 9;
        break;
      }
    case 8:
      {
        time_t_now++;
        if (time_t_now >= 21 * NNF) _start_send = 10;
        break;
      }
    case 9:
      {
        time_t_now++;
        if (time_t_now >= 7 * NNF) _start_send = 10;
        break;
      }
    case 10:
      {
        _bit_i = _bit_i + 1;
        if (_bit_i >= 32) _start_send = 11;
        else _start_send = 5;
        break;
      }
    case 11:
      {
        updateDC(50);
        time_t_now = 0;
        _start_send = 12;
        break;
      }
    case 12:
      {
        time_t_now++;
        if (time_t_now >= 7 * NNF) _start_send = 13;
        break;
      }
    case 13:
      {
        updateDC(0);
        _start_send = 0;
        break;
      }

    default:
      {
        break;
      }
  }
}

void in1_handler() {
  int l;
  l = digitalRead(IN1);
  int long t0;
  t0 = Timer0Count;
  int long dt = t0 - t1;

  if (dt > 200) {
    state = 0;
    i = 0;
  }
  switch (state) {
    case 0:
      {
        if ((l == 1) && (dt > 120)) state = 1;
        break;
      }
    case 1:
      {
        if ((l == 0) && (dt > 60)) state = 2;
        i = 0;
        break;
      }
    case 2:
      {
        ls[i] = l;
        ts[i] = dt;
        i++;
        if (i >= 64) {
          i = 0;
          state = 3;
        }
        break;
      }
    default:
      {
        break;
      }
  }
  t1 = t0;
}




/************VLC Add************/
// Function to calculate checksum (simple XOR)
uint8_t calculateChecksum(uint8_t* data, size_t length) {
    uint8_t checksum = 0;
    for (size_t i = 0; i < length; i++) {
        checksum ^= data[i];
    }
    return checksum;
}

void DecodeReceivedData(){
if (state == 3) {
    int j;
    int address = 0x0000, naddress = 0x0000;
    int data = 0x0000, ndata = 0x0000;
    for (j = 0; j < 64; j = j + 2) {
      //Serial.print(ls[j]); Serial.print(","); Serial.print(ts[j]); Serial.print("|");
      //Serial.print(abs(ts[j]-ts[j+1])<10?0:1);
      switch (j >> 4) {
        case 0:
          {
            address |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 1:
          {
            naddress |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 2:
          {
            data |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 3:
          {
            ndata |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            //Serial.print((abs(ts[j]-ts[j+1])<10?0:1));
            break;
          }
        default:
          {
            break;
          }
      }
    }
    if (~(data & ndata) && ~(address & naddress)) {
      dataReady = true;
      receivedData = data;
      // Serial.print("NBVLC RX: Addr:");
      // Serial.print(address);
      // Serial.print(", Data:");
      // Serial.println(data);
    }
    state = 0;
    collected_data.spare_3  = collected_data.spare_3  + 4;
    //set 7 bit. maybe impact vlc to receive the data?
    collected_data.running_mode |= (1 << 7); 
    send_periodic_frame_example(true); //send node status
  }


}

// States for the frame receiver
enum FrameState {
    WAIT_START,
    RECEIVE_HEADER,
    RECEIVE_DATA,
    RECEIVE_CHECKSUM,
    WAIT_END,
    FRAME_COMPLETE
};
FrameState currentState = WAIT_START; // Initial state
uint8_t dataLength = 0;  // Length of the data field
uint8_t calculatedChecksum = 0; // Calculated checksum

void printBufferHex(uint8_t* buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if (buffer[i] < 0x10) Serial.print("0"); // Add leading zero for single-digit hex
    Serial.print(buffer[i], HEX);           // Print each byte in hexadecimal
    Serial.print(" ");                      // Add a space between bytes
  }
  Serial.println();                         // Add a newline at the end
}

void processFrame() {
    DecodeReceivedData();
    if (dataReady) {
        dataReady = false;
        // Serial.print(receivedData, HEX);
        // Serial.print(" ");
        switch (currentState) {
            case WAIT_START:
                if (receivedData == START_FRAME) {
                    frameIndex = 0;
                    frameBuffer[frameIndex++] = receivedData;
                    currentState = RECEIVE_HEADER;
                }
                else{
                    frameIndex = 0;
                    currentState = WAIT_START;
                }
                break;

            case RECEIVE_HEADER:
                frameBuffer[frameIndex++] = receivedData;
                // After receiving the address (2 bytes), command, and data length
                if (frameIndex == 5) {
                    // Check address
                    uint8_t Destination_Address =  frameBuffer[2];
                    if (Destination_Address != ADDRESS_Source) {
                        Serial.println("Error: Address mismatch");
                        frameIndex = 0;
                        currentState = WAIT_START;
                        break;
                    }
                    // Store data length
                    dataLength = frameBuffer[4];

                    // Transition to RECEIVE_DATA or directly to RECEIVE_CHECKSUM if no data
                    if (dataLength > 0) {
                        currentState = RECEIVE_DATA;
                    } else {
                        currentState = RECEIVE_CHECKSUM;
                    }
                }
                break;

            case RECEIVE_DATA:
                frameBuffer[frameIndex++] = receivedData;

                // Transition to checksum state when all data bytes are received
                if (frameIndex == 5 + dataLength) {
                    currentState = RECEIVE_CHECKSUM;
                }
                break;

            case RECEIVE_CHECKSUM:
                frameBuffer[frameIndex++] = receivedData;
                // Calculate and verify checksum
                calculatedChecksum = calculateChecksum(&frameBuffer[0], frameIndex - 2);
                if (calculatedChecksum != receivedData) {
                    Serial.println("Error: Checksum mismatch");
                    currentState = WAIT_START;
                    break;
                }

                currentState = WAIT_END;
                break;

            case WAIT_END:
                if (receivedData == END_FRAME) {
                    frameBuffer[frameIndex++] = receivedData;
                    currentState = FRAME_COMPLETE; // Successfully received frame
                    frameIndex_clone = frameIndex;
                    frameIndex = 0;
                } else {
                    Serial.println("Error: Missing end frame");
                    currentState = WAIT_START;
                    frameIndex = 0;
                }
                break;

            default:
                currentState = WAIT_START; // Reset in case of unexpected state
                frameIndex = 0;
                break;
        }

        // Prevent buffer overflow
        if (frameIndex >= sizeof(frameBuffer)) {
            Serial.println("Error: Frame too large");
            currentState = WAIT_START;
        }
    }

    if (currentState == FRAME_COMPLETE) {
        currentState = WAIT_START;
        frameFlag = true;  // Set flag indicating a valid frame is received
        frameIndex = 0;
        Serial.println("Frame received successfully");
    }

    // CLear bit 7 . maybe impact vlc to receive the data?
    collected_data.running_mode &= ~(1 << 7);
    send_periodic_frame_example(true); //send node status
}

void encodeAndStartSend(uint8_t c) {
    nbvlc_on();  //added 09/09/25
    _c = ((uint32_t)c << 16) | ((~(uint32_t)c) << 24) | 0x0000FF00;
    _start_send = 1;  // Start transmission state machine
    // while(_start_send != 0) {  // Wait for transmission to complete
    //     delay(1);
    // }
    nbvlc_off();  //added 09/09/25
    
    collected_data.vlc_communication_volume += 4;
    collected_data.vlc_information_volume = collected_data.vlc_communication_volume - 3;
}

 

    
void queueFrameForSend(uint8_t* data, size_t len) {
    if (frameSending || len > FRAME_BUFFER_SIZE) return; // Skip if already sending or too large

    
    memcpy((void*)frameQueue, data, len); // Copy to global queue
    frameQueueLen = len;
    frameSendIndex = 0;
    frameSending = true;

     encodeAndStartSend(frameQueue[0]);
     frameSending = false; //added 09/09/25
}

// Function to send frame
void sendFrame(uint8_t addr) {
    uint8_t frame[64]; // Adjust size as needed
    size_t index = 0;
    // Start of frame
    frame[index++] = START_FRAME;

    // Address (2 bytes)
    frame[index++] = ADDRESS_Source;
    frame[index++] = addr;

    // Command
    frame[index++] = CMD_SEND;


    // Perform a reading to update sensor values
    bool readingSuccessful = false; // Flag to track if reading was successful
    for (size_t attempt = 0; attempt < 5; attempt++) {
      
      nbvlc_off(); //added 09/09/25
      
        if (bme.performReading()) {
            readingSuccessful = true; // Reading succeeded
            Serial.println("Reading sensor succefully!");
            break; // Exit the loop
        } else {
            // Reading failed, attempt to reconfigure the sensor
            Serial.print("Attempting to reconfigure sensor (Attempt ");
            Serial.print(attempt + 1);
            Serial.println(" of 5)");
            bme_setup(); // Reconfigure the sensor
            if (attempt >= 5)
            {
              Serial.print("Reading failed. Have tried to setup sensor multiple times.");
            }
        }
    }

    if (!readingSuccessful) {
        // If all attempts fail, print an error message and exit
        Serial.println("Failed to perform reading from BME680 sensor after 5 attempts!");
        return;
    }
    

    // Data (temperature, pressure, humidity, gas_resistance, altitude)
    float data[] = {
        bme.temperature,
        (float)bme.pressure/ 100.0,
        bme.humidity,
        (float)bme.gas_resistance/ 1000.0,
        bme.readAltitude(SEALEVELPRESSURE_HPA)
    };

    size_t dataSize = sizeof(data) / sizeof(data[0]);
    // Number of data bytes
    frame[index++] = dataSize * sizeof(float);

    // Copy actual data (convert float to bytes)
    for (size_t i = 0; i < dataSize; i++) {
        uint8_t* dataBytes = (uint8_t*)&data[i];
        for (size_t j = 0; j < sizeof(float); j++) {
            frame[index++] = dataBytes[j];
        }
    }
    // Checksum
    frame[index++] = calculateChecksum(&frame[0], index - 1);

    // End of frame
    frame[index++] = END_FRAME;

    // // Send the frame using send_NEC
    // Serial.print("Frame: ");
    // for (size_t i = 0; i < index; i++) {
    //     while (_start_send) {
    //         // Wait until the previous byte is sent
    //         delay(300); // Small delay to allow for sending time
    //     }
    //   send_NEC(frame[i]);
    //   // Print the frame byte in hexadecimal
    //   Serial.print(frame[i], HEX);
    //   Serial.print(" ");
    // }
    // Serial.println(" ");
    // Serial.println("Send successfully");

    // Serial.print("Queuing frame: ");
    // for (size_t i = 0; i < index; i++) {
    //     Serial.print(frame[i], HEX);
    //     Serial.print(" ");
    // }
    // Serial.println();
    //set 1 bit, reprsent sending
    collected_data.running_mode |= (1 << 1); 
    send_periodic_frame_example(true);
    send_periodic_frame_example(true);
    
    queueFrameForSend(frame, index);
    
}

void test_nbvlc_frame() {
Serial.println("Prepare frame");
sendFrame(ADDRESS_Destination); // Send data

delay(100);
}
/************END VLC ************/







/************BLE Add************/

void onRXWrite(uint16_t conn_handle, BLECharacteristic* chr, uint8_t* data, uint16_t len) {
    // Add new data to the circular buffer
    for (uint16_t i = 0; i < len; i++) {
        BLE_frameBuffer[BLE_bufferHead] = data[i];
        BLE_bufferHead = (BLE_bufferHead + 1) % BLE_FRAME_BUFFER_SIZE;

        // Check for buffer overflow
        if (BLE_bufferHead == BLE_bufferTail) {
            Serial.println("Frame buffer overflow! Discarding old data.");
            BLE_bufferTail = (BLE_bufferTail + 1) % BLE_FRAME_BUFFER_SIZE;  // Discard oldest data
        }
    }
    collected_data.spare_2 = collected_data.spare_2 + len;//count rx
    // Attempt to process frames from the buffer
    BLE_processFrames();
}

void BLE_processFrames() {
    // Ensure the buffer contains at least the minimum frame size
    //set 7 bit. maybe impact vlc to receive the data?
    collected_data.running_mode |= (1 << 6); 
    send_periodic_frame_example(true); //send node status
    send_periodic_frame_example(true); //send node status
    send_periodic_frame_example(true); //send node status
    send_periodic_frame_example(true); //send node status
    Serial.print("Received BLE Frame: ");
    for (size_t i = 0; i < 11; i++) {
      // Print the frame byte in hexadecimal
      Serial.print(BLE_frameBuffer[BLE_bufferTail+i], HEX);
      Serial.print(" ");
    }
    Serial.println(" ");

    while ((BLE_bufferTail != BLE_bufferHead)&(ble_frameFlag==false)){
        // Check for the start of a frame
        if (BLE_frameBuffer[BLE_bufferTail] != START_FRAME) {
            BLE_bufferTail = (BLE_bufferTail + 1) % BLE_FRAME_BUFFER_SIZE;  // Skip invalid data
            continue;
        }
        // if (BLE_frameBuffer[BLE_bufferTail+2] != ADDRESS_Source) {
        //     BLE_bufferTail = (BLE_bufferTail + 1) % BLE_FRAME_BUFFER_SIZE;  // Skip invalid data
        //     continue;
        // }
        // Ensure the buffer contains at least the minimum frame size
        size_t bufferLength = (BLE_bufferHead >= BLE_bufferTail)
                              ? (BLE_bufferHead - BLE_bufferTail)
                              : (BLE_FRAME_BUFFER_SIZE - BLE_bufferTail + BLE_bufferHead);

        if (bufferLength < 11) {  // Minimum frame: START + 6xADDR + CMD + LENGTH + CS + END
            return;  // Wait for more data
        }
        // Read the length field
        uint8_t length = BLE_frameBuffer[(BLE_bufferTail + 8) % BLE_FRAME_BUFFER_SIZE];
        size_t frameSize = 11 + length;  // Calculate full frame size

        // Check if the entire frame is available in the buffer
        if (bufferLength < frameSize) {
            return;  // Wait for more data
        }
        // Validate the end of the frame
        uint8_t endIndex = (BLE_bufferTail + frameSize - 1) % BLE_FRAME_BUFFER_SIZE;
        if (BLE_frameBuffer[endIndex] != END_FRAME) {
            BLE_bufferTail = (BLE_bufferTail + 1) % BLE_FRAME_BUFFER_SIZE;  // Skip invalid frame
            continue;
        }
        // Calculate and validate the checksum
        uint8_t calculatedChecksum = calculateChecksum(&BLE_frameBuffer[BLE_bufferTail], frameSize - 3); // Exclude cs_FRAME and END_FRAME
        uint8_t receivedChecksum = BLE_frameBuffer[(BLE_bufferTail + frameSize - 2) % BLE_FRAME_BUFFER_SIZE];
        // Serial.print("calculatedChecksum");
        // Serial.println(calculatedChecksum, HEX);
        // Serial.print("receivedChecksum");
        // Serial.println(receivedChecksum, HEX);
        if (calculatedChecksum != receivedChecksum) {
            Serial.println("Invalid checksum, discarding frame.");
            BLE_bufferTail = (BLE_bufferTail + 1) % BLE_FRAME_BUFFER_SIZE;  // Skip invalid frame
            continue;
        }
        // Extract and process the frame payload
        // processFramePayload(BLE_bufferTail, frameSize);
        BLE_frameSize = frameSize;
        ble_frameFlag = true;
        Serial.print("Received one frame successfully!");
        // Remove the processed frame from the buffer
        // BLE_bufferTail = (BLE_bufferTail + frameSize) % BLE_FRAME_BUFFER_SIZE;
        // memset(BLE_frameBuffer, 0, BLE_FRAME_BUFFER_SIZE);
    }

    // CLear bit 1 . maybe impact ble to receive the data?
    collected_data.running_mode &= ~(1 << 6);
    send_periodic_frame_example(true); //send node status
    Serial.flush();   // ensure serial data fully sent
    send_periodic_frame_example(true); //send node status
    Serial.flush();   // ensure serial data fully sent
    send_periodic_frame_example(true); //send node status
    Serial.flush();   // ensure serial data fully sent
    send_periodic_frame_example(true); //send node status
    Serial.flush();   // ensure serial data fully sent
}




BLE_Data processFramePayload(size_t start, size_t frameSize) {
    uint8_t addr_ble[6]={0};
    size_t index = (start + 1) % BLE_FRAME_BUFFER_SIZE;  // Skip START_FRAME
    addr_ble[0] = BLE_frameBuffer[index];
    addr_ble[1] = BLE_frameBuffer[(index + 1) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[2] = BLE_frameBuffer[(index + 2) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[3] = BLE_frameBuffer[(index + 3) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[4] = BLE_frameBuffer[(index + 4) % BLE_FRAME_BUFFER_SIZE];
    addr_ble[5] = BLE_frameBuffer[(index + 5) % BLE_FRAME_BUFFER_SIZE];

    uint8_t command = BLE_frameBuffer[(index + 6) % BLE_FRAME_BUFFER_SIZE];
    uint8_t length = BLE_frameBuffer[(index + 7) % BLE_FRAME_BUFFER_SIZE];

    // if (command == CMD_SEND)
    // {
    //   // Extract payload data
    //   uint8_t payload[length];
    //   for (uint8_t i = 0; i < length; i++) {
    //       payload[i] = BLE_frameBuffer[(index + 8 + i) % BLE_FRAME_BUFFER_SIZE];
    //   }
    //   // Print or handle the received frame
    //   // Serial.print("Received Frame: ");
    //   // Serial.print("Header=0x");
    //   // Serial.print(BLE_frameBuffer[index-1]);
    //   // for (uint8_t i=0; i<6; i++)
    //   // {
    //   //   Serial.print(", Address=0x");
    //   //   Serial.print(addr_ble[i], HEX);
    //   // }

    //   // Serial.print(", Command=0x");
    //   // Serial.print(command, HEX);
    //   // Serial.print(", Payload=");
    //   // for (uint8_t i = 0; i < length; i++) {
    //   //     Serial.print(payload[i], HEX);
    //   //     Serial.print(" ");
    //   // }
    //   // Serial.println();
    // }
    // else if (command == CMD_RECEIVE)//upload sensing to gateway
    // {
    //   send_ble_frame(addr_ble);
    // }

    BLE_Data data = {{addr_ble[0], addr_ble[1], addr_ble[2], addr_ble[3], addr_ble[4], addr_ble[5]}, command, length, index};
    return data;
}


void send_ble_frame(uint8_t* addr) {
    uint8_t frame[64]; // Adjust size as needed
    size_t index = 0;

    // Start of frame
    frame[index++] = START_FRAME;

    frame[index++] = addr[0]; // BLE_ADDRESS 0
    frame[index++] = addr[1]; // BLE_ADDRESS 1 
    frame[index++] = addr[2]; // BLE_ADDRESS 2
    frame[index++] = addr[3]; // BLE_ADDRESS 3
    frame[index++] = addr[4]; // BLE_ADDRESS 4
    frame[index++] = addr[5]; // BLE_ADDRESS 5

    // Command
    frame[index++] = CMD_SEND;
    // Perform a reading to update sensor values
    bool readingSuccessful = false; // Flag to track if reading was successful

    for (size_t attempt = 0; attempt < 5; attempt++) {
    if (bme.performReading()) {
        readingSuccessful = true; // Reading succeeded
        break; // Exit the loop
    } else {
        // Reading failed, attempt to reconfigure the sensor
        Serial.print("Reading failed. Attempting to reconfigure sensor (Attempt ");
        Serial.print(attempt + 1);
        Serial.println(" of 5)");
        bme_setup(); // Reconfigure the sensor
    }
  }


    if (!readingSuccessful) {
        // If all attempts fail, print an error message and exit
        Serial.println("Failed to perform reading from BME680 sensor after 5 attempts!");
        return;
    }
    

    // Data (temperature, pressure, humidity, gas_resistance, altitude)
    float data[] = {
        bme.temperature,
        (float)bme.pressure/ 100.0,
        bme.humidity,
        (float)bme.gas_resistance/ 1000.0,
        bme.readAltitude(SEALEVELPRESSURE_HPA)
    };

    size_t dataSize = sizeof(data) / sizeof(data[0]);
    // Number of data bytes
    frame[index++] = dataSize * sizeof(float);

    // Copy actual data (convert float to bytes)
    for (size_t i = 0; i < dataSize; i++) {
        uint8_t* dataBytes = (uint8_t*)&data[i];
        for (size_t j = 0; j < sizeof(float); j++) {
            frame[index++] = dataBytes[j];
        }
    }

    // Checksum
    frame[index++] = calculateChecksum(&frame[0], index - 1);

    // End of frame
    frame[index++] = END_FRAME;

    // Send the frame using send_NEC
    bool result = charTX.notify(frame, index);
    // If sending was successful, update the global bytes counter
    if(!result) {
      collected_data.ble_communication_volume += index;

      collected_data.ble_information_volume = collected_data.ble_communication_volume - 11;
      //set 0 bit, reprsent sending
      collected_data.running_mode |= (1 << 0); 
      send_periodic_frame_example(true);
      Serial.flush();   // ensure serial data fully sent
      send_periodic_frame_example(true);
      Serial.flush();   // ensure serial data fully sent
      send_periodic_frame_example(true);
      Serial.flush();   // ensure serial data fully sent
      // Serial.print("The number of bytes: ");
      // Serial.print(collected_data.ble_communication_volume);
    }


    // Serial.print("Frame: ");
    // for (size_t i = 0; i < index; i++) {
    //   // Print the frame byte in hexadecimal
    //   Serial.print(frame[i], HEX);
    //   Serial.print(" ");
    // }

    // Serial.println(" ");
    // Serial.println("Send ble frame successfully");
}

void test_ble_frame() {
Serial.println("Prepare BLE frame");
send_ble_frame(BLE_ADDRESS); // Send data
delay(100);
}
/************END BLE Add************/



unsigned long advertisingStartTime;


int TxPower = -4;//TX=-20dBm, -16dBm, -12dBm, -8dBm, -4dBm, 0dBm, +3dBm and +4dBm.
int advertising_interval = int(250/0.625); //unit is 0.645fast:20-100ms, slow:100-2000ms. if set up 20ms, it is equal to 20/0.625
int Connection_Interval = int(45/1.25); // unit is 1.25ms, if set up 45ms, it is equal to 45/1.25

bool inFastMode = true;
int Advertising_fast_time = 30;

int phy_rate  = PHY_1M; //1Mbps, 2Mbps, 500 kbps/125 kbps
int mtu_value = 31; //23-247bytes


// void setup() {
//   Serial.begin(115200);
//   while (!Serial) delay(10);  // for nrf52833 with native usb 

//   Serial.println("SUPERIOT test FW v. 1.0 2023-2024");
//   Serial.println("-----------------------");



//   // Config the peripheral connection with maximum bandwidth
//   // more SRAM required by SoftDevice
//   // Note: All config***() function must be called before begin()
  
//   Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);  //  CONTROLS MTU SIZE  //***************
//   // Initialise the Bluefruit module
//   Bluefruit.begin();
//   //nbvlc_off();

//   Bluefruit.setTxPower(0);  // Set transmit power to 0 dBm 

//   Bluefruit.setName("[SUPERIOT] Node #005");

//   // Set the connect/disconnect callback handlers
//   Bluefruit.Periph.setConnectCallback(connect_callback);
//   Bluefruit.Periph.setDisconnectCallback(disconnect_callback);

//   Bluefruit.Periph.setConnInterval(36, 36); // 6 - 7.5 - 36 - 45 ms   //***************

//   // Configure and Start the Device Information Service
//   bledis.setManufacturer("SUPERIOT");
//   bledis.setModel("Node Core");
//   bledis.setHardwareRev("2.0");
//   bledis.setSoftwareRev("1.0");
//   bledis.begin();

//   // Start the BLE Battery Service and set it to 100%
//   blebas.begin();
//   blebas.write(100);

       
//  // Setup the myService and myChar
//  setup_myChar();

        
// // Setup the advertising packet(s)  
//   startAdv(collected_data.fast_time);

//   //----
//   //bme_setup();
//   //as_setup();
//   //bme_print();

//   // Interval in microsecs
//   if (ITimer0.attachInterruptInterval(TIMER0_INTERVAL_MS / 10, TimerHandler0)) {
//     Serial.print(F("Starting ITimer0 OK, millis() = "));
//     Serial.println(millis());
//   } else
//     Serial.println(F("Can't set ITimer0. Select another freq. or timer"));

//   pinMode(pinToUse, OUTPUT);
//   pinMode(IN1, INPUT_PULLUP);
//   digitalWrite(pinToUse, LOW);

//   attachInterrupt(digitalPinToInterrupt(IN1), in1_handler, CHANGE);
//   PWM_Instance = new nRF52_PWM(pinToUse, frequency, dutyCycle);

//   pinMode(18, INPUT);  //WAKWUP input
//   pinMode(5, OUTPUT);  // EPD POWER ON
//   digitalWrite(5, HIGH);

//   nbvlc_off(); //??? not work
//   Serial.println("Ready and advertising");
// }



void setup() {
  collected_data.running_state = 0;
  Serial.begin(115200);
  send_periodic_frame_example(true);
  while (!Serial) delay(10);  // for nrf52833 with native usb  

  Serial.println("SUPERIOT test FW v. 1.0 2023-2024");
  Serial.println("-----------------------");


  // initize the structure
    // collected_data.running_state             = 0;
    collected_data.wakeup_state           = 1;
    collected_data.communication_mode     = 0;
    collected_data.running_mode           = 0;

    collected_data.vlc_protocol_mode      = 2;
    collected_data.vlc_interval_seconds   = TIMER0_INTERVAL_MS / 10;
    collected_data.vlc_communication_volume = 0;
    collected_data.vlc_information_volume = 0;
    collected_data.vlc_per_unit_time      = 20;
    collected_data.pwm_percentage         = 50;

    collected_data.ble_protocol_mode      = 2;
    collected_data.ble_interval_seconds   = Connection_Interval; //max( (uint16_t)6, min( (uint16_t)3200, (uint16_t)(collected_data.ble_interval_seconds / 1.25f + 0.5f) ) ); // units: 1.25ms
    collected_data.ble_communication_volume = 0;
    collected_data.ble_information_volume = 0;
    collected_data.ble_per_unit_time      = 10;
    collected_data.phy_rate_percentage    = phy_rate;
    collected_data.mtu_value              = mtu_value; //The range is from 23 to 247.
    collected_data.fast_time              = Advertising_fast_time; //The range is from 23 to 247.

    collected_data.spare_2 = 0; 
    collected_data.spare_3 = 0;
    collected_data.spare_4 = 0;


  // Config the peripheral connection with maximum bandwidth
  // more SRAM required by SoftDevice
  // Note: All config***() function must be called before begin()
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);  // ADDED THROUGHPUT - CONTROLS MTU SIZE //**********************************************************************

  // Initialise the Bluefruit module
  Bluefruit.begin();
 
  Bluefruit.setTxPower(TxPower);  // Set transmit power to 0 dBm //********************************************************************************************************************************
  collected_data.spare_4 |= (1 << 2); //bit 2 becomes 1  TX=-20dBm, -16dBm, -12dBm, -8dBm, -4dBm, 0dBm, +3dBm and +4dBm.
  collected_data.spare_4 |= (1 << 4); //bit 4 becomes 1, 0b00010100

  // val &= ~(1 << 5);          // bit 5 becomes 0
  // val |= (1 << 5);           // bit 5 becomes 1
  /************Add************/
  // Format the node name with the source address
  char nodeName[20]; // Buffer to hold the formatted string
  snprintf(nodeName, sizeof(nodeName), "[SUPIOT] Node #%03d", ADDRESS_Source);
  // Set the name
  Bluefruit.setName(nodeName);
  /************ Add************/
  // Bluefruit.setName("[SUPERIOT] Node #001");

  // Set the connect/disconnect callback handlers
  Bluefruit.Periph.setConnectCallback(connect_callback);
  Bluefruit.Periph.setDisconnectCallback(disconnect_callback);
  Bluefruit.Periph.setConnInterval(36, 36); // 6 - 7.5 - 36 - 45 ms               

 //**************BLE add**************//
  customService.begin();
  // Configure RX characteristic (Write Without Response)
  charRX.setProperties(CHR_PROPS_WRITE_WO_RESP);
  charRX.setPermission(SECMODE_OPEN, SECMODE_OPEN);
  charRX.setWriteCallback(onRXWrite); // Set write callback for incoming frames
  charRX.begin();

  // myService.begin();
  // myChar.setProperties(CHR_PROPS_READ | CHR_PROPS_WRITE_WO_RESP);
  // myChar.setPermission(SECMODE_OPEN, SECMODE_OPEN);
  // myChar.setFixedLen(1);  // Alternatively .setMaxLen(uint16_t len)
  // myChar.setWriteCallback(myChar_write_cb);
  // myChar.begin();

  // Configure TX characteristic (Notify)
  charTX.setProperties(CHR_PROPS_NOTIFY);
  charTX.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  charTX.setFixedLen(collected_data.mtu_value);  // Optional: Fixed length for the frame size
  charTX.setCccdWriteCallback(cccd_callback); // Attach CCCD callback
  charTX.begin();
    // Add the service
  //Bluefruit.Periph.setConnInterval(36, 36); // 6 - 7.5 - 36 - 45 ms             
  // Configure and Start the Device Information Service
  bledis.setManufacturer("SUPERIOT");
  bledis.setModel("Node Core");
  bledis.setHardwareRev("2.0");
  bledis.setSoftwareRev("1.0");
  bledis.begin();

  // Start the BLE Battery Service and set it to 100%
  blebas.begin();
  blebas.write(100);

  // Setup the myService and myChar
  // setup_myChar();

  // Setup the advertising packet(s)  
  startAdv(collected_data.fast_time);
  collected_data.running_state = 1;
  advertisingStartTime = millis();
  inFastMode = true;
  collected_data.spare_4 &= ~(1 << 0);   // bit 0 becomes 0

  //  // Clear bit 5
  // val &= ~(1 << 5);          // bit 5 becomes 0
  // // Set bit 5
  // val |= (1 << 5);           // bit 5 becomes 1


  //bme_setup();
  //as_setup();

  // Interval in microsecs
  if (ITimer0.attachInterruptInterval(TIMER0_INTERVAL_MS / 10, TimerHandler0)) {
    Serial.print(F("Starting ITimer0 OK, millis() = "));
    Serial.println(millis());
  } else
    Serial.println(F("Can't set ITimer0. Select another freq. or timer"));

  pinMode(pinToUse, OUTPUT);
  pinMode(IN1, INPUT_PULLUP);
  digitalWrite(pinToUse, LOW);

  attachInterrupt(digitalPinToInterrupt(IN1), in1_handler, CHANGE);
  PWM_Instance = new nRF52_PWM(pinToUse, frequency, dutyCycle);

  pinMode(18, INPUT);  //WAKWUP input
  pinMode(5, OUTPUT);  // EPD POWER ON
  digitalWrite(5, HIGH);

  // pinMode(22, OUTPUT); // Power to VLC RX
  // digitalWrite(22, HIGH);  

  // Serial.println("Ready and advertising");
  nbvlc_off();
  // nbvlc_on();


}

#define CHECK_INTERVAL_MS 10000L
#define CHANGE_INTERVAL_MS 20000L

void updateDC(uint16_t level) {
  uint32_t _level;
  // Mapping data to any other frequency from original data 0-100 to actual 16-bit Dutycycle
  PWM_Instance->setPWM_manual(pinToUse, ((uint32_t)level * MAX_16BIT) / 100);
}

void send_NEC(int8_t c) {
  _c = ((uint32_t)c) << 16 | (~(uint32_t)c) << 24 | 0x0000ff00;
  // Serial.println(_c);
  _start_send = 1;
}

void setup_myChar() {
  myService.begin();
  myChar.setProperties(CHR_PROPS_READ | CHR_PROPS_WRITE_WO_RESP);
  myChar.setPermission(SECMODE_OPEN, SECMODE_OPEN);
  myChar.setFixedLen(1);  // Alternatively .setMaxLen(uint16_t len)
  myChar.setWriteCallback(myChar_write_cb);
  myChar.begin();

  bleuart.begin();

  // Set callback for received data
  bleuart.setRxCallback(bleRxCallback); // transmitting sensor data via BLE

//*************************** ADDED THROUGHPUT SENSOR TEXT BLE***************************
// bleuart.begin();
// //bleuart.setRxCallback(bleuart_rx_callback); // used for throughput test
// bleuart.setNotifyCallback(bleuart_notify_callback);
// // Set callback for received data
// bleuart.setRxCallback(bleRxCallback); // transmitting sensor data via BLE
//***************************
}


void startAdv(int fast_time) {
  // Advertising packet
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.ScanResponse.addName();

  // Bluefruit.Advertising.addService(bleuart);
  // Include HRM Service UUID
  Bluefruit.Advertising.addService(myService);
  // Bluefruit.Advertising.addService(customService);
  // Bluefruit.Advertising.addService(charRX);  // Include RX characteristic in the advertising
  // Bluefruit.Advertising.addService(charTX);  // Include TX characteristic in the advertising
  Bluefruit.Advertising.addService(bleuart);

  // Include Name
  Bluefruit.Advertising.addName();

  /* Start Advertising
   * - Enable auto advertising if disconnected
   * - Interval:  fast mode = 20 ms, slow mode = 152.5 ms
   * - Timeout for fast mode is 30 seconds
   * - Start(timeout) with timeout = 0 will advertise forever (until connected)
   * 
   * For recommended advertising interval
   * https://developer.apple.com/library/content/qa/qa1931/_index.html   
   */
  Bluefruit.Advertising.restartOnDisconnect(true);
  // Bluefruit.Advertising.setInterval(32, 244);  // in unit of 0.625 ms //************************************************************************************************
  Bluefruit.Advertising.setInterval(advertising_interval, advertising_interval);
  collected_data.spare_4 &= ~(1 << 0);   // fast mode, bit 0 becomes 0
  //  // Clear bit 5
  // val &= ~(1 << 5);          // bit 5 becomes 0
  // // Set bit 5
  // val |= (1 << 5);           // bit 5 becomes 1
  Bluefruit.Advertising.setFastTimeout(fast_time);  // number of seconds in fast mode
  Bluefruit.Advertising.start(0);            // 0 = Don't stop advertising after n seconds
  collected_data.communication_mode = 1;
}

//****************************************ADDED *************************************************8
void setConnectionInterval(uint16_t conn_handle, uint16_t min_interval, uint16_t max_interval) {
  ble_gap_conn_params_t conn_params;
  conn_params.min_conn_interval = min_interval;  // Minimum connection interval (1.25ms units)
  conn_params.max_conn_interval = max_interval;  // Maximum connection interval (1.25ms units)
  conn_params.slave_latency = 0;                 // Slave latency
  conn_params.conn_sup_timeout = 500;            // Supervision timeout (10ms units)
  // Apply the connection parameters
  sd_ble_gap_conn_param_update(conn_handle, &conn_params);
}
// *********************************************************************************************************8


void connect_callback(uint16_t conn_handle) {
  Bluefruit.setTxPower(TxPower);  // Adjust this value as needed //***************************************************************************************************************
  collected_data.communication_mode = 2;
  // Get the reference to current connection
  BLEConnection* connection = Bluefruit.Connection(conn_handle);
  //***************************************************************************************** 
  setConnectionInterval(conn_handle, Connection_Interval, Connection_Interval);  // Example values (100ms to 200ms) //************************ 
 //**********************************************************************************************************
  char central_name[32] = { 0 };
  connection->getPeerName(central_name, sizeof(central_name));
  Serial.print("Connected to ");
  Serial.println(central_name);

  // *************************************************************************************************************
  //nbvlc_off(); // **********************************************ADDED TO REDUCE POWER DURING BLE CONNECTION MODE
  //**********************************************************************************************************
  
  //****************************************THROUGHPUT ***************************************************
  // // request PHY changed to 2MB
  //   Serial.println("Request to change PHY");
  //   connection->requestPHY();

  //   // request to update data length
  //   Serial.println("Request to change Data Length");
  //   connection->requestDataLengthUpdate();

  //   // request mtu exchange
  //   Serial.println("Request to change MTU");
  //   connection->requestMtuExchange(247);

  //  // request connection interval of 7.5 ms
  //  connection->requestConnectionParameter(36); // in unit of 1.25 //****************************************

  //   // delay a bit for all the request to complete
  //   delay(1000);

  // *************************************************************************************************************
  // nbvlc_off(); // **********************************************TO REDUCE POWER DURING BLE CONNECTION MODE
  //**********************************************************************************************************
}

/**
 * Callback invoked when a connection is dropped
 * @param conn_handle connection where this event happens
 * @param reason is a BLE_HCI_STATUS_CODE which can be found in ble_hci.h
 */
void disconnect_callback(uint16_t conn_handle, uint8_t reason) {
  (void)conn_handle;
  (void)reason;
  collected_data.communication_mode = 1;

  // Serial.print("Disconnected, reason = 0x");
  // Serial.println(reason, HEX);
  // Serial.println("Advertising!");

  //****************************************THROUGHPUT ***************************************************
  //  test_running = false; // Stop the test if disconnected
  //  test_completed = false; // Reset test completion flag
  //*********************************************************************************************************************
}

void cccd_callback(uint16_t conn_hdl, BLECharacteristic* chr, uint16_t cccd_value) {
  // Display the raw request packet
  Serial.print("CCCD Updated: ");
  //Serial.printBuffer(request->data, request->len);
  Serial.print(cccd_value);
  Serial.println("");

  //**********Add************//
  if (cccd_value & BLE_GATT_HVX_NOTIFICATION) {
    Serial.println("Notifications enabled");
  } else {
    Serial.println("Notifications disabled");
  }

  // Check the characteristic this CCCD update is associated with in case
  // this handler is used for multiple CCCD records.
  //if (chr->uuid == hrmc.uuid) {
  //    if (chr->notifyEnabled(conn_hdl)) {
  //        Serial.println("Heart Rate Measurement 'Notify' enabled");
  //    } else {
  //        Serial.println("Heart Rate Measurement 'Notify' disabled");
  //    }
  //}
}

void CMD_BLE_Intervel_Process()
{
  if (_conn_handle != BLE_CONN_HANDLE_INVALID) {
        BLEConnection* connection = Bluefruit.Connection(_conn_handle);

        if (connection && connection->connected()) {
          bool update_requested_this_loop = false; 
          // --- Check and Request Connection Interval Update ---
          uint16_t desired_interval_units = max( (uint16_t)6, min( (uint16_t)3200, (uint16_t)(collected_data.ble_interval_seconds / 1.25f + 0.5f) ) ); // units: 1.25ms

          if (desired_interval_units != last_requested_interval_units) {
            Serial.print("Interval Preference Changed. Requesting: ");
            Serial.print(desired_interval_units * 1.25f); Serial.print(" ms (Units: ");
            Serial.print(desired_interval_units); Serial.println(")");

            if (connection->requestConnectionParameter(desired_interval_units)) {
              last_requested_interval_units = desired_interval_units;
              update_requested_this_loop = true;
            } else {
              Serial.println("--> Failed to send Interval change request.");
            }
          }
      }else if (connection && !connection->connected()) {
        // Connection object exists but is no longer connected (should be handled by disconnect_callback, but good safety check)
        _conn_handle = BLE_CONN_HANDLE_INVALID;
      }
  }
}

void CMD_PHY_Rate_Process()
{
  int desired_phy_pref;         // 1 for 1M, 2 for 2M, 3 for Coded
  if (collected_data.phy_rate_percentage == PHY_1M) { // High value on param1 -> High throughput
    desired_phy_pref = PHY_1M;
  }else if (collected_data.phy_rate_percentage == PHY_2M) {
    desired_phy_pref = PHY_2M; // Or PHY_CODED if you need long range
  }else if (collected_data.phy_rate_percentage == PHY_3M) {
    desired_phy_pref = PHY_3M; // Or PHY_CODED if you need long range
  }
    else { 
    desired_phy_pref = PHY_1M; 
    collected_data.phy_rate_percentage = PHY_1M;
  }

  // ---  Parameter Update Logic ---
  if (_conn_handle != BLE_CONN_HANDLE_INVALID) {
    // We are connected
    BLEConnection* connection = Bluefruit.Connection(_conn_handle);

    // Ensure connection object is still valid
    if (connection && connection->connected()) {

      bool param_update_requested = false;

      // --- Check and Request PHY Update ---
      if (desired_phy_pref != last_requested_phy_pref) {
        Serial.print("PHY preference changed. Requesting: ");
        uint8_t req_phy_mask = 0;
        switch(desired_phy_pref) {
            case PHY_1M:    Serial.println("1M"); req_phy_mask = BLE_GAP_PHY_1MBPS; break;
            case PHY_2M:    Serial.println("2M"); req_phy_mask = BLE_GAP_PHY_2MBPS; break;
            case PHY_3M: Serial.println("Coded (Long Range)"); req_phy_mask = BLE_GAP_PHY_CODED; break;
            default:        Serial.println("Default/Auto"); req_phy_mask = BLE_GAP_PHY_AUTO; break; // Should not happen with good logic
        }

        // Request symmetric PHY (same for TX and RX)
        // You can specify different PHYs for tx & rx if needed: requestPHY(tx_phy, rx_phy, options)
        if (connection->requestPHY(req_phy_mask)) {
            last_requested_phy_pref = desired_phy_pref; // Update last requested value *only if request was sent*
            param_update_requested = true;
        } else {
            Serial.println("Failed to *send* PHY change request.");
        }
      }

      else if (connection && !connection->connected()) {
      // Connection object exists but is no longer connected (should be handled by disconnect_callback, but good safety check)
      _conn_handle = BLE_CONN_HANDLE_INVALID;
      }
    }
  }
}

void CMD_MTU_Process()
{
  uint16_t desired_mtu;         // Desired MTU
  if (_conn_handle != BLE_CONN_HANDLE_INVALID) {
    BLEConnection* connection = Bluefruit.Connection(_conn_handle);

    if (connection && connection->connected()) {
      bool update_requested_this_loop = false; 
      
      desired_mtu = max((uint16_t)23, min((uint16_t)247, collected_data.mtu_value)); // Clamp MTU

      collected_data.mtu_value = desired_mtu;

      if (desired_mtu != last_requested_mtu) {
      Serial.print("MTU Preference Changed. Requesting: "); Serial.println(desired_mtu);
      Serial.println("--> Also requesting Data Length Update (DLE).");
      connection->requestDataLengthUpdate(); // Good practice with MTU change

      if (connection->requestMtuExchange(desired_mtu)) {
        last_requested_mtu = desired_mtu;
        update_requested_this_loop = true;
      } else {
        Serial.println("--> Failed to send MTU change request.");
      }
    }
  }else if (connection && !connection->connected()) {
    // Connection object exists but is no longer connected (should be handled by disconnect_callback, but good safety check)
    _conn_handle = BLE_CONN_HANDLE_INVALID;
  }

  }

}


unsigned long lastResetTime = millis();
// Function to report bytes sent in the last 10 seconds and reset the counter
void reportAndResetBytesCount() {
  // Serial.print("Bytes sent in the last unit seconds: ");
  // Serial.println(collected_data.ble_communication_volume);
  
  // Reset the counter
  collected_data.ble_communication_volume = 0;
  collected_data.ble_information_volume = 0;
  lastResetTime = millis();
  send_periodic_frame_example(true);
}

unsigned long lastResetTime_RX = millis();
// Function to report bytes sent in the last 10 seconds and reset the counter
void reportAndResetBytesCount_RX() {
  // Serial.print("Bytes sent in the last unit seconds: ");
  // Serial.println(collected_data.ble_communication_volume);
  
  // Reset the RX counter
  collected_data.spare_2 = 0;
  lastResetTime_RX = millis();
  send_periodic_frame_example(true);
}


void wakeup_process()
{
  processFrame();
  if (frameBuffer[3] == CMD_Wakeup)
  {
    collected_data.wakeup_state = frameBuffer[4];
    if (collected_data.wakeup_state == 1)
    {
      bme_setup();
      as_setup();
      nbvlc_on();
      epd_message();
      Serial.println("Wake up !");
    }
  }


}

void deep_sleep_process()
{
    collected_data.running_state = 3;
    send_periodic_frame_example(true);
    delay(20);
    send_periodic_frame_example(true); //send node status
    Serial.println("deep sleep !");
    // bme_setup();
    as_setup();
    // nbvlc_off();
    // epd.Sleep();
    delay(10);
    systemOff(18, 1);
}


//AA vlc-ble cmd data1 data2 data3 ... data n EE. E.g., AA 02 04  1e 11 21 EE


//////////////////////ble/////////////////
  // unsigned long vlc_sent = 0;
  static bool auto_send_vlc_flag;

void vlc_sendPeriodicFrames(uint8_t address, unsigned long interval_s, unsigned long count) {
  static unsigned long vlc_nextSendTime = 0; // only set once
  static unsigned long vlc_sent = 0;

  if (vlc_sent < count) {
    unsigned long now = millis();
    if (now >= vlc_nextSendTime) {
      nbvlc_on();
      // sendFrame(address);
      test_nbvlc_rx_tx(0x13);
      vlc_sent++;
      vlc_nextSendTime = now + interval_s * 1000UL;  // schedule next send

      // Serial.print("[Done] vlc_Sent ");
      // Serial.print(vlc_sent);
      // Serial.println(" frames");
    }
  } else {
    // delay(12000);
    delay(100);
    send_periodic_frame_example(true);
    // CLear bit 1 
    collected_data.running_mode &= ~(1 << 1);
    send_periodic_frame_example();
    auto_send_vlc_flag = false;
    vlc_sent = 0;              // reset for next use
    vlc_nextSendTime = 0;      // reset timer
    nbvlc_off();
  }
}


static bool vlc_running = false;
static unsigned long vlc_interval_s;
static uint8_t vlc_address;
static unsigned long vlc_duration_s;


void auto_send_vlc()
{
  if (auto_send_vlc_flag==true) 
  {
    vlc_sendPeriodicFrames(vlc_address, vlc_interval_s, vlc_duration_s);
  }

}

void handleVLC(uint8_t cmd, uint8_t* data1)
{
  if (cmd == CMD_RECEIVE)
  {
    sendFrame(data1[0]); // Send data to destination gateway
    Serial.println("Send frame to gateway !");
    delay(100);
  }

  else if (cmd == CMD_Running_State)
  {
    collected_data.running_state = data1[0];
    send_periodic_frame_example(true); //send node status
  }
  else if (cmd == CMD_Wakeup)
  {
    collected_data.wakeup_state = data1[0];
    if (collected_data.wakeup_state == 0)// go to sleep
    {
      deep_sleep_process();
    }
  }
  else if (cmd == CMD_Communcation_mode)
  {
    collected_data.communication_mode = data1[0];
    send_periodic_frame_example(true); //send node status
  }
  else if (cmd == CMD_VLC_Protocol)
  {
    collected_data.vlc_protocol_mode = data1[0]; 
    if ((collected_data.vlc_protocol_mode != 1)|(collected_data.vlc_protocol_mode != 2))
    {
      collected_data.vlc_protocol_mode == 1;
    }
    send_periodic_frame_example(true); //send node status

  }
  else if (cmd == CMD_VLC_Intervel)
  {
    collected_data.vlc_interval_seconds = data1[0]; 
    if (collected_data.vlc_interval_seconds>=5000)
    {
      collected_data.vlc_interval_seconds = 5000;
    }
    else if (collected_data.vlc_interval_seconds<=1)
    {
    collected_data.vlc_interval_seconds = 1;
    }
    ITimer0.setInterval(int(collected_data.vlc_interval_seconds / 10), TimerHandler0);  
    send_periodic_frame_example(true); //send node status        
  }
  else if (cmd == CMD_PWM)
  {
    collected_data.pwm_percentage = data1[0];  
    send_periodic_frame_example(true); //send node status          
  }
  else if (cmd == CMD_VLC_Unit)
  {
    collected_data.vlc_per_unit_time = data1[0];
    send_periodic_frame_example(true); //send node status
    Serial.print("vlc_per_unit_time:");
    Serial.println(collected_data.vlc_per_unit_time);  
  }
  else if (cmd == CMD_VLC_SEND_Auto)//const uint8_t CMD_VLC_SEND_Auto = 0x10; //SEND VLC AUTO FOR TEST //e.g., AA 01 10 05 20 01 EE
  {
    vlc_interval_s = data1[0];
    vlc_duration_s = data1[1];
    vlc_address   = data1[2];
    auto_send_vlc_flag = true; 
  }
  else if (cmd == CMD_VLC_OFF_ON)//set vlc off 0 or on 1
  {
    if (data1[0] == 0)
    {
      nbvlc_off(); 
    // CLear bit 3 
      collected_data.running_mode &= ~(1 << 3);
      send_periodic_frame_example(true); //send node status
      Serial.print("VLC OFF:");
    }
    else if(data1[0] == 1)
    {
      nbvlc_on(); 
      collected_data.running_mode |= (1 << 3); 
      send_periodic_frame_example(true); //send node status
      Serial.print("VLC ON:");
    }
  }
  else if (cmd == CMD_EINK_OFF_ON)//set vlc off 0 or on 1
  {
    if (data1[0] == 0)
    {
      // epd.Sleep(); //consump more energy??
      // CLear bit 4 
      // collected_data.running_mode &= ~(1 << 4);
      Serial.print("EINK OFF:");
    }
    else if(data1[0] == 1)
    { // Set bit 4 of running_mode
      collected_data.running_mode |= (1 << 4); 
      send_periodic_frame_example(true);
      Serial.print("EINK Display.");
      epd_message();
      // CLear bit 4 
      collected_data.running_mode &= ~(1 << 4);
      send_periodic_frame_example(true);
      Serial.print("End EINK Display");
    }
  }
  else
  {
    Serial.print("Input value is invalid, set EINK fail:");
  }
}


//////////////////////ble/////////////////

static bool auto_send_ble_flag;


void ble_sendPeriodicFrames(uint8_t* address, unsigned long interval_s, unsigned long count) {
  static unsigned long ble_nextSendTime = 0; // only set once
  static unsigned long ble_sent = 0;

  if (ble_sent < count) {
    unsigned long now = millis();
    if (now >= ble_nextSendTime) {
      send_ble_frame(address);
      ble_sent++;
      ble_nextSendTime = now + interval_s * 1000UL;  // schedule next send

      // Serial.print("[Done] ble_sent ");
      // Serial.print(ble_sent);
      // Serial.println(" frames");
    }
  } else {
    delay(100);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    // CLear bit 1 
    collected_data.running_mode &= ~(1 << 0);
    send_periodic_frame_example(true);
    Serial.flush();   // ensure serial data fully sent
    auto_send_ble_flag = false;
    ble_sent = 0;              // reset for next use
    ble_nextSendTime = 0;      // reset timer
  }
}

static bool ble_running = false;
static unsigned long ble_interval_s;
static uint8_t ble_address[6] = {0xD6, 0xD6, 0x70, 0x4D, 0x29, 0x6B};
static unsigned long ble_duration_s;


void auto_send_ble()
{
  if (auto_send_ble_flag==true) 
  {
    ble_sendPeriodicFrames(ble_address, ble_interval_s, ble_duration_s);
  }

}


unsigned long state2_advertisingStartTime;
void handleBLE(uint8_t cmd, uint8_t* data1)
{

    if (cmd == CMD_RECEIVE)
    {
      send_ble_frame(data1);
    }

    else if (cmd == CMD_Running_State)
    {
      collected_data.running_state = data1[0];
      send_periodic_frame_example(true); //send node status
    }
    else if (cmd == CMD_Wakeup)
    {
      if (collected_data.wakeup_state == 0)// go to sleep
      {
        collected_data.wakeup_state = data1[0];
        Serial.print("BLE wakeup_state:");
        Serial.println(collected_data.wakeup_state);
        deep_sleep_process();
      }
    }
    else if (cmd == CMD_Communcation_mode)
    {
      collected_data.communication_mode = data1[0];
      send_periodic_frame_example(true); //send node status
    }
    else if (cmd == CMD_BLE_Protocol)
    {
      collected_data.ble_protocol_mode = data1[0]; 
      send_periodic_frame_example(true); //send node status           
    }
    else if (cmd == CMD_BLE_Intervel)
    {
      collected_data.ble_interval_seconds = data1[0];
      Serial.print("ble_interval_seconds:");
      Serial.println(collected_data.ble_interval_seconds);
      CMD_BLE_Intervel_Process();
      send_periodic_frame_example(true); //send node status

    }
    else if (cmd == CMD_PHY_Rate)
    {
      collected_data.phy_rate_percentage = data1[0];
      Serial.print("phy_rate_percentage:");
      Serial.println(collected_data.phy_rate_percentage);  
      CMD_PHY_Rate_Process(); 
      send_periodic_frame_example(true);//send node status
    }
    else if (cmd == CMD_MTU)
    {
        collected_data.mtu_value = data1[0];
        Serial.print("mtu_value:");
        Serial.println(collected_data.mtu_value); 
        CMD_MTU_Process();
        send_periodic_frame_example(true); //send node status
    }
    else if (cmd == CMD_BLE_Unit)
    {
      collected_data.ble_per_unit_time = data1[0];
      Serial.print("ble_per_unit_time:");
      Serial.println(collected_data.ble_per_unit_time); 
      send_periodic_frame_example(true); //send node status 
    }
    else if (cmd == CMD_BLE_SEND_Auto)//const uint8_t CMD_BLE_SEND_Auto = 0x20; //SEND BLE AUTO FOR TEST  //e.g., AA 02 20 05 20 D6 D6 70 4D 29 6B EE
    {
      ble_interval_s = data1[0];
      ble_duration_s = data1[1];
      ble_address[0]   = data1[2];
      ble_address[1]   = data1[3];
      ble_address[2]   = data1[4];
      ble_address[3]   = data1[5];
      ble_address[4]   = data1[6];
      ble_address[5]   = data1[7];
      auto_send_ble_flag = true; 
    }
    else if (cmd == CMD_BLE_OFF_ON)
    {
      if (data1[0] == 0)
      {
        // Mode 0: BLE Off
        Bluefruit.Advertising.stop();
        sd_softdevice_disable();  // Nordic SDK call to fully turn off BLE
        Serial.println("BLE fully disabled");
        // CLear bit 2 
        collected_data.running_mode &= ~(1 << 2);
        send_periodic_frame_example(true); //send node status
      }
      else if(data1[0] == 1)
      {
        // Mode 1: BLE Idle (enabled but not advertising)
        Bluefruit.begin();  // Re-enable BLE if it was off
        Bluefruit.Advertising.stop();
        Serial.println("BLE enabled, not advertising");
        // Set bit 2 of running_mode
        collected_data.running_mode |= (1 << 2); 
        send_periodic_frame_example(true); //send node status
      }
    }
    else if (cmd == CMD_BLE_FAST_TIME)
    {
      if(Bluefruit.connected())
      {
        // Mode: BLE advertising
        collected_data.fast_time = data1[0];
        startAdv(collected_data.fast_time);
        state2_advertisingStartTime = millis();
        Serial.print("Advertising, ble_fast_time:");
        Serial.println(collected_data.fast_time);  
        // Set bit 5 of running_mode
        collected_data.running_mode |= (1 << 5); 
        send_periodic_frame_example(true); //send node status
      }
      else
      {
        Serial.print("Ble is connected, setup failed.");
      }

    }
}






void serial_processing() {
  // 1) wait for a full ASCII‑hex line ending in '\n'
  if (! Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  // 2) split on spaces, hex→bytes
  uint8_t frame[MAX_TOKENS];
  size_t  flen = 0;
  int     idx  = 0;

  while (idx < line.length() && flen < MAX_TOKENS) {
    int sp = line.indexOf(' ', idx);
    String tok;
    if (sp == -1) {
      tok = line.substring(idx);
      idx = line.length();
    } else {
      tok = line.substring(idx, sp);
      idx = sp + 1;
    }
    tok.trim();
    if (tok.length() == 0) continue;

    // convert token ("AA","01",...) to a byte
    char buf[3] = {0};
    tok.toCharArray(buf, 3);
    frame[flen++] = (uint8_t) strtoul(buf, nullptr, 16);
  }

  // 3) validate minimum length & markers
  //    need at least START, link, cmd, END → flen>=4
  if (flen < 4 ||
      frame[0] != START_FRAME ||
      frame[flen-1] != END_FRAME)
  {
    Serial.println(F("✗ Invalid frame; expect AA vlc_ble cmd [data...] EE"));
    return;
  }

  // 4) extract link & cmd
  uint8_t link = frame[1];   // 1 = VLC, 2 = BLE
  uint8_t cmd  = frame[2];

  // 5) compute data count & pointer
  size_t dataCount = flen - 4;       // subtract START, link, cmd, END
  uint8_t* dataPtr = &frame[3];      // first data byte

  // 5) dispatch & print
  switch (link) {
    case 1:
      Serial.print(F("▶ VLC command: 0x"));
      if (cmd   < 0x10) Serial.print('0');
      Serial.print(cmd,   HEX);
      handleVLC(cmd, dataPtr);
      break;

    case 2:
      Serial.print(F("▶ BLE command: 0x"));
      if (cmd   < 0x10) Serial.print('0');
      Serial.print(cmd,   HEX);
      handleBLE(cmd, dataPtr);
      break;

    case 3:
      Serial.print(F("▶ J VLC command: 0x"));
      if (cmd   < 0x10) Serial.print('0');
      Serial.print(cmd,   HEX);
      // processReceivedData(cmd, dataPtr);
      processSerialCommand(cmd, dataPtr);
      break;

    default:
      Serial.print(F("✗ Unknown link type: "));
      Serial.println(link, DEC);
      break;
  }




  // // print the data bytes
  // Serial.print(F("    data:"));
  // for (size_t i = 0; i < dataCount; i++) {
  //   Serial.print(' ');
  //   if (dataPtr[i] < 0x10) Serial.print('0');
  //   Serial.print(dataPtr[i], HEX);
  // }
  // Serial.println();
}



unsigned long vlc_lastResetTime = millis();
// Function to report bytes sent in the last 10 seconds and reset the counter
void vlc_reportAndResetBytesCount() {
  // Serial.print("vlc Bytes sent in the last unit seconds: ");
  // Serial.println(collected_data.vlc_communication_volume);
  
  // Reset the counter
  collected_data.vlc_communication_volume = 0;
  collected_data.vlc_information_volume = 0;
  vlc_lastResetTime = millis();
  send_periodic_frame_example(true); //send node status
}

unsigned long vlc_lastResetTime_RX = millis();
// Function to report bytes sent in the last 10 seconds and reset the counter
void vlc_reportAndResetBytesCount_RX() {
  // Serial.print("vlc RX Bytes sent in the last unit seconds: ");
  // Serial.println(vlc_lastResetTime_RX);

  //   Serial.print("millis(): ");
  // Serial.println(millis());
  
  // Reset the counter
  collected_data.spare_3 = 0;
  vlc_lastResetTime_RX = millis();
  send_periodic_frame_example(true); //send node status
}

void loop()
{
    serial_processing();// serial command processing
    auto_send_ble();
    auto_send_vlc();

    if (collected_data.running_state == 1 && inFastMode) {
    if ((millis() - advertisingStartTime) > (Advertising_fast_time*1000)) {
      inFastMode = false;
      collected_data.running_state = 2;
      send_periodic_frame_example(true); //send node status
      send_periodic_frame_example(true); //send node status
      Serial.println("End advertising and state 1, switched to state 2.");
    }
    }

    if ((collected_data.running_mode & 1) == 0)
    {
      collected_data.ble_communication_volume = 0;
      collected_data.ble_information_volume = 0;
    }

    if ((collected_data.running_mode & 0b10) == 0)
    {
      collected_data.vlc_communication_volume = 0;
      collected_data.vlc_information_volume = 0;
    }


    if (collected_data.running_state == 2 && (collected_data.running_mode & (1 << 5))) {
    if ((millis() - state2_advertisingStartTime) > (collected_data.fast_time*1000)) {
      collected_data.fast_time = 0;
      collected_data.running_mode &= ~(1 << 5);//clear 5 bit
      send_periodic_frame_example(true); //send node status
      Serial.println("End advertising.");
    }
    }

  // Check if unit seconds have elapsed; millis() returns time in milliseconds
  if ((millis() - lastResetTime) >= (collected_data.ble_per_unit_time*1000)) {
    reportAndResetBytesCount();
  }

  if ((millis() - lastResetTime_RX) >= (collected_data.ble_per_unit_time*1000)) {
    reportAndResetBytesCount_RX();
  }

    // Check if unit seconds have elapsed; millis() returns time in milliseconds
  if ((millis() - vlc_lastResetTime) >= (collected_data.vlc_per_unit_time*1000)) {
    vlc_reportAndResetBytesCount();
  }

  if ((millis() - vlc_lastResetTime_RX) >= (collected_data.vlc_per_unit_time*1000)) {
    vlc_reportAndResetBytesCount_RX();
  }

  send_periodic_frame_example(); //send node status

  delay(100);  

}

// Create and send frame
Frame frame;

bool processSerialCommand(uint8_t cmd, uint8_t* data1) {
      switch(cmd) {
          case 0x06:
              frame.command  = cmd;  // Change this to any command type you need
              break;
          case 0x07:
              frame.command = cmd;  // Change this to any command type you need
              break;
          case 0x08:
              frame.command = cmd;  // Change this to any command type you need
              break;
          default:
              Serial.println("Error: Invalid command. Valid commands are:");
              Serial.println("06 = Request Sensor Data");
              Serial.println("07 = Request Location Data");
              Serial.println("08 = Request Both Data");
              return false;
      }
            
      // nbvlc_off();
      //  bme_setup();
      //  bme_print();
      float temperature = bme.temperature;
      float pressure = bme.pressure / 100.0;
      float humidity = bme.humidity;
      float gas = bme.gas_resistance / 1000.0;
      //  epd_message();

      // Prepare sensor and location data
      SensorData sensorData = prepareSensorData(temperature, pressure, humidity, gas);

      // Define current location coordinates
      float x_coord = 9.23;  // X coordinate in meters
      float y_coord = -4.56; // Y coordinate in meters
      float z_coord = 9.89;  // Z coordinate in meters

      LocationData locationData = prepareLocationData(x_coord, y_coord, z_coord);


      frame.startMarker  = START_MARKER;
      frame.sourceAddr   = MY_ADDRESS;
      frame.destAddr     = data1[0];
      // frame.command      = CMD_COMBINED;  // Change this to any command type you need

    uint8_t* payloadPtr = frame.payload;
    // Automatically set payload size and copy data based on command type
    switch(frame.command) {
        case CMD_REQ_SENSOR:
            frame.payloadSize = sizeof(SensorData);
            memcpy(payloadPtr, &sensorData, sizeof(SensorData));
            break;
        case CMD_REQ_LOC:
            frame.payloadSize = sizeof(LocationData);
            memcpy(payloadPtr, &locationData, sizeof(LocationData));
            break;
        case CMD_REQ_BOTH:
            frame.payloadSize = sizeof(SensorData) + sizeof(LocationData);
            memcpy(payloadPtr, &sensorData, sizeof(SensorData));
            payloadPtr += sizeof(SensorData);
            memcpy(payloadPtr, &locationData, sizeof(LocationData));
            break;
        case CMD_ACK:
        case CMD_NACK:
        default:
            frame.payloadSize = 0;  // No payload for request/ack/nack commands
            break;
      }

      frame.checksum    = calculateChecksum(&frame);
      frame.endMarker   = END_MARKER;
      // nbvlc_on();
      // Transmit the frame
      transmitFrame(frame);
      return true;
}





// void loop() {
//   // serial_processing();// serial command processing
//   // auto_send_ble();
//   // auto_send_vlc();

//     if (collected_data.running_state == 1 && inFastMode) {
//     if ((millis() - advertisingStartTime) > (Advertising_fast_time*1000)) {
//       inFastMode = false;
//       collected_data.running_state = 2;
//       send_periodic_frame_example(true); //send node status
//       Serial.println("End advertising and state 1, switched to state 2.");
//     }
//     }

//     if ((collected_data.running_mode & 1) == 0)
//     {
//       collected_data.ble_communication_volume = 0;
//       collected_data.ble_information_volume = 0;
//     }

//     if ((collected_data.running_mode & 0b10) == 0)
//     {
//       collected_data.vlc_communication_volume = 0;
//       collected_data.vlc_information_volume = 0;
//     }


//     if (collected_data.running_state == 2 && (collected_data.running_mode & (1 << 5))) {
//     if ((millis() - state2_advertisingStartTime) > (collected_data.fast_time*1000)) {
//       collected_data.fast_time = 0;
//       collected_data.running_mode &= ~(1 << 5);//clear 5 bit
//       send_periodic_frame_example(true); //send node status
//       Serial.println("End advertising.");
//     }
//     }

//   // Check if unit seconds have elapsed; millis() returns time in milliseconds
//   if ((millis() - lastResetTime) >= (collected_data.ble_per_unit_time*1000)) {
//     reportAndResetBytesCount();
//   }

//   if ((millis() - lastResetTime_RX) >= (collected_data.ble_per_unit_time*1000)) {
//     reportAndResetBytesCount_RX();
//   }

//     // Check if unit seconds have elapsed; millis() returns time in milliseconds
//   if ((millis() - vlc_lastResetTime) >= (collected_data.vlc_per_unit_time*1000)) {
//     vlc_reportAndResetBytesCount();
//   }

//   if ((millis() - vlc_lastResetTime_RX) >= (collected_data.vlc_per_unit_time*1000)) {
//     vlc_reportAndResetBytesCount_RX();
//   }

//   send_periodic_frame_example(); //send node status

//   if (collected_data.wakeup_state == 0)// now is sleeping , node is waiting for wakeup signal
//   {
//     wakeup_process();
//   }
//   else if (collected_data.wakeup_state == 1) //default is 1, wake up
//   {
//     /*****************VLC**********************/ 
//     if ((collected_data.communication_mode == 2)|(collected_data.communication_mode == 1)|(collected_data.communication_mode == 0)) //Only VLC OR BOTH VLC/BLE
//     {
//       if (collected_data.vlc_protocol_mode == 1) //bit level protocol
//       {
//         // processReceivedData(); 
//         ;
//       }
//       else if (collected_data.vlc_protocol_mode == 2) //byte level protocol
//       {
//         processFrame();
//         if (frameFlag) 
//         {
//             if (frameBuffer[3] == CMD_RECEIVE)
//             {
//               sendFrame(frameBuffer[1]); // Send data to destination gateway
//               Serial.println("Send frame to gateway !");
//               delay(100);
//             }

//             else if (frameBuffer[3] == CMD_Running_State)
//             {
//               collected_data.running_state = frameBuffer[4];
//             }
//             else if (frameBuffer[3] == CMD_Wakeup)
//             {
//               collected_data.wakeup_state = frameBuffer[4];
//               if (collected_data.wakeup_state == 0)// go to sleep
//               {
//                 deep_sleep_process();
//               }
//             }
//             else if (frameBuffer[3] == CMD_Communcation_mode)
//             {
//               collected_data.communication_mode = frameBuffer[4];
//             }
//             else if (frameBuffer[3] == CMD_VLC_Protocol)
//             {
//               collected_data.vlc_protocol_mode = frameBuffer[4]; 
//               if ((collected_data.vlc_protocol_mode != 1)|(collected_data.vlc_protocol_mode != 2))
//               {
//                 collected_data.vlc_protocol_mode == 1;
//               }

//             }
//             else if (frameBuffer[3] == CMD_VLC_Intervel)
//             {
//               collected_data.vlc_interval_seconds = frameBuffer[4]; 
//               if (collected_data.vlc_interval_seconds>=5000)
//               {
//                 collected_data.vlc_interval_seconds = 5000;
//               }
//               else if (collected_data.vlc_interval_seconds<=1)
//               {
//                collected_data.vlc_interval_seconds = 1;
//               }
//               ITimer0.setInterval(int(collected_data.vlc_interval_seconds / 10), TimerHandler0);          
//             }
//             else if (frameBuffer[3] == CMD_PWM)
//             {
//               collected_data.pwm_percentage = frameBuffer[4];            
//             }
//              else if (frameBuffer[3] == CMD_VLC_Unit)
//             {
//               collected_data.vlc_per_unit_time = frameBuffer[4];
//               Serial.print("vlc_per_unit_time:");
//               Serial.println(collected_data.vlc_per_unit_time);  
//             }
//             frameIndex_clone = 0;  // Reset frameIndex_clone
//             frameFlag = false;  // Reset the flag
//         }
//       }
//     }

//     /*****************BLE**********************/ 
//     if ((collected_data.communication_mode == 2)|(collected_data.communication_mode == 1)|(collected_data.communication_mode == 0)) //Only BLE OR BOTH BLE/VLC
//     {
//       if (collected_data.ble_protocol_mode == 1) //bit level protocol
//       {
//         if (ble_frameFlag) {
//           ble_frameFlag = false;  // Reset the flag
//         }
//       }
//       else if (collected_data.ble_protocol_mode == 2) //byte level protocol
//       {
//         if (ble_frameFlag) 
//         {
//             BLE_Data ble_received_data = processFramePayload(BLE_bufferTail, BLE_frameSize);
//             if (ble_received_data.command == CMD_RECEIVE)
//             {
//               send_ble_frame(ble_received_data.addr_ble_rec);
//             }

//             else if (ble_received_data.command == CMD_Running_State)
//             {
//               collected_data.running_state = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//             }
//             else if (ble_received_data.command == CMD_Wakeup)
//             {
//               collected_data.wakeup_state = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//               Serial.print("BLE wakeup_state:");
//               Serial.println(collected_data.wakeup_state);
//               if (collected_data.wakeup_state == 0)// go to sleep
//               {
//                 deep_sleep_process();
//               }
//             }
//             else if (ble_received_data.command == CMD_Communcation_mode)
//             {
//               collected_data.communication_mode = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//             }
//             else if (ble_received_data.command == CMD_BLE_Protocol)
//             {
//               collected_data.ble_protocol_mode = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];            
//             }
//             else if (ble_received_data.command == CMD_BLE_Intervel) // uint16_t desired_interval_units = max( (uint16_t)6, min( (uint16_t)3200, (uint16_t)(collected_data.ble_interval_seconds / 1.25f + 0.5f) ) ); // units: 1.25ms
//             {
//               collected_data.ble_interval_seconds = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//               Serial.print("ble_interval_seconds:");
//               Serial.println(collected_data.ble_interval_seconds);
//               CMD_BLE_Intervel_Process();

//             }
//             else if (ble_received_data.command == CMD_PHY_Rate)
//             {
//               collected_data.phy_rate_percentage = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//               Serial.print("phy_rate_percentage:");
//               Serial.println(collected_data.phy_rate_percentage);  
//               CMD_PHY_Rate_Process(); 
//             }
//             else if (ble_received_data.command == CMD_MTU)
//             {
//                 collected_data.mtu_value = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//                 Serial.print("mtu_value:");
//                 Serial.println(collected_data.mtu_value); 
//                 CMD_MTU_Process();
//             }
//             else if (ble_received_data.command == CMD_BLE_Unit)
//             {
//               collected_data.ble_per_unit_time = BLE_frameBuffer[(ble_received_data.index + 8) % BLE_FRAME_BUFFER_SIZE];
//               Serial.print("ble_per_unit_time:");
//               Serial.println(collected_data.ble_per_unit_time);  
//             }

//             BLE_bufferTail = (BLE_bufferTail + BLE_frameSize) % BLE_FRAME_BUFFER_SIZE;
//             memset(BLE_frameBuffer, 0, BLE_FRAME_BUFFER_SIZE);
//             ble_frameFlag = false;  // Reset the flag 
//         }
//       }    


//     }
//   }

// }


/***************** moved all subtest functions in main.c to this function**********************/ 
void test_all()
{

if (Bluefruit.connected()) {
    //Test cases triggered by myChar write/update
    switch (char_m) {
      case 1:
        {
          Serial.println("Test mode #1 activated (eink and sensor test).");
          bme_setup();
          as_setup();
          nbvlc_off();  //******************************************* ADDED TO REDUCE CURRENT IN FIRMWARE TEST MODE 01******************************
          char_m = 0;
          tm = 1;
          break;
        }
      case 2:
        {
          Serial.println("Test mode #2 activated (wakeup test).");
          Serial.println("Use NBVLC signal or an IR remote to generate a wakeup interrupt form AS3933.");
          char_m = 0;
          tm = 2;
          bme_setup();
          as_setup();
          nbvlc_off();
          //sleep();
          break;
        }
      case 3:
        {
          Serial.println("Test mode #3 activated (NBVLC rx and tx test).");
          char_m = 0;
          nbvlc_on();
          tm = 3;
          break;
        }
      case 6:
        {
          Serial.println("Test mode #6 activated (NBVLC presence/ok).");
          char_m = 0;
          nbvlc_on();
          tm = 6;
          break;
        }
      case 7:
        {
          Serial.println("Test mode #7 activated (NBVLC presence/error).");
          char_m = 0;
          nbvlc_on();
          tm = 7;
          break;
        }
      case 8:  //  Add for Send one frame via vlc
        {
          Serial.println("Test mode #8 activated (Send one frame via vlc).");
          char_m = 0;
          nbvlc_on();
          tm = 8;
          break;
        }
      case 9:  //  Add for Send one frame via ble
        {
          Serial.println("Test mode #9 activated (Send one frame via ble).");
          char_m = 0;
          tm = 9;
          break;
        }
      case 10:
        {
          Serial.println("Test mode #10 activated (eink image for show).");
          char_m = 0;
          tm = 10;
          break;
        }

        // ****************************************** SENSOR TX ********************************************
        case 15: {
          Serial.println("Test mode #15 activated (Sensor readings transmission via BLE and data reception).");
          Serial.println("Setup complete, sending data...");
          bme_setup();
          as_setup();          
          nbvlc_off();
          char_m = 0;
          tm = 15;
          break;
        }
        // ******************************************THROUGHPUT****************************************************
        case 16: {
        Serial.println("Test mode #16 activated (Throughput Test).");
        as_setup();
        nbvlc_off();
        char_m = 0;
        tm = 16;
        break;
        }

        // *******************************SENSOR, E-ink and NBVLC in low power mode (nbvlc happens 2s after e-ink (due to delay of 2s) and next sensing happens 5 s after nbvlc)********************************************************
      case 17:
        {
          Serial.println("Test mode #17 activated (low power: eink, sensor and nbvlc test).");
          bme_setup();
          as_setup();
          nbvlc_off();  
          char_m = 0;
          tm = 17;
          break;
        }
        // *******************************SENSOR, E-ink and NBVLC in normal mode (nbvlc happens 2s after e-ink (due to delay of 2s) and next sensing happens 5 s after nbvlc)********************************************************
      case 18:
        {
          Serial.println("Test mode #18 activated (normal mode: eink, sensor and nbvlc test).");
          bme_setup();
          as_setup();
          char_m = 0;
          tm = 18;
          break;
        }
        // *********************************************************************************************
        case 19:
        {
          Serial.println("Test mode #19 activated (normal mode: sensor and nbvlc test).");
          bme_setup();
          as_setup();
          char_m = 0;
          tm = 19;
          break;
        }
      // *********************************************************************************************
        case 20:
        {
          Serial.println("Test mode #20 activated (low power mode: sensor and nbvlc test).");
          bme_setup();
          as_setup();
          nbvlc_off();  
          char_m = 0;
          tm = 20;
          break;
        }
      // 
      // *********************************************************************************************
        case 21:
        {
          Serial.println("Test mode #21 activated (low power mode: sensor test).");
          bme_setup();
          as_setup();
          nbvlc_off();  
          char_m = 0;
          tm = 21;
          break;
        }
      //**************************************************************************************************
        {
          Serial.println("Test mode #22 activated (low power: Eink, sensor and nbvlc sensor TX test)."); // CONSIDERING VLC FRAME TX/RX  
          bme_setup();
          as_setup();
          //nbvlc_off();             
          char_m = 0;
          tm = 22;
          break;
        }

      // *********************************************************************************************
      default: break;
    }
  }
  //branches based on current test mode
  switch (tm) {
    case 1:
      {
        test_eink_and_sensor();
        break;
      }
    case 2:
      {
        test_wakeup();
        break;
      }
    case 3:
      {
        test_nbvlc_rx_tx(0x11);
        break;
      }
    case 6:
      {
        test_nbvlc_rx_tx(0x12);  //presence with ok
        break;
      }
    case 7:
      {
        test_nbvlc_rx_tx(0x13);  //presence with error
        break;
      }
    case 8:
      {
        //test_nbvlc_rx_tx(0x11); //no presence
        test_nbvlc_frame(); 
        tm = 0;
        break;
      }
    case 9:
      {
        //test_nbvlc_rx_tx(0x11); //no presence
        test_ble_frame(); 
        tm = 0;
        break;
      }
    case 10:
      {
        epd_image();  //no presence
        tm = 0;
        break;
      }
      // ***************************** SENSOR TX *********************
      // case 15: {
      // //BLE_TX_sensor();
      // test_eink_and_sensor_ble_tx_sensor() ;
      // //tm=0;
      // break;
      // }

      // ******************THROUGHPUT***********************
      // case 16: {
      //   if (Bluefruit.connected() && bleuart.notifyEnabled())
      // {
      // checkAndRunTest();
      // }
      // break;
      // }
      // ****************************************************************

      // ******************sensor, e-ink, and nbvlc ***********************
    case 17:
      {
        test_eink_and_sensor_nbvlc(); // lowpower
        break;
      }
      // ************************************************************************

    case 18:
      {
        test_eink_and_sensor_nbvlc_normal();
        break;
      }
      // ************************************************************************
      case 19:
      {
        test_sensor_nbvlc_normal();
        break;
      }
      //**************************************************************************
      case 20:
      {
        test_sensor_nbvlc_lowpower();
        break;
      }
      //**************************************************************************
      case 21:
      {
        test_sensor_lowpower();
        break;
      }
      //**************************************************************************
    case 22:
      {
        //test_eink_and_sensor_nbvlc_TX_readings();  
        test_eink_and_sensor_nbvlc_TX_readings_2();
        break;
      }

    default: break;
  }

  // Only send update once per 100 mseconds
  
   delay(100);  // ***************************** COMMENETED FOR THROUGHPUT TEST  *********************


}



int test_delay_ctr = 0;

void test_eink_and_sensor() {
  if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
    test_delay_ctr = 0;
    bme_print();
    //delay(2000); //****************************************************************************************
    epd_message();
  }
}

void test_wakeup() {
  if (digitalRead(18) == HIGH) {
    Serial.println("WAKEUP!");
    bme_print();
    while (digitalRead(18) == HIGH) asTag.clear_wake();
    Serial.println("Going to SLEEP.");
    delay(1000);
  }
}

void test_nbvlc_rx_tx(unsigned char c) {
  if (state == 3) {
    int j;
    int address = 0x0000, naddress = 0x0000;
    int data = 0x0000, ndata = 0x0000;

    for (j = 0; j < 64; j = j + 2) {
      //Serial.print(ls[j]); Serial.print(","); Serial.print(ts[j]); Serial.print("|");
      //Serial.print(abs(ts[j]-ts[j+1])<10?0:1);
      switch (j >> 4) {
        case 0:
          {
            address |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 1:
          {
            naddress |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 2:
          {
            data |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 3:
          {
            ndata |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            //Serial.print((abs(ts[j]-ts[j+1])<10?0:1));
            break;
          }
        default:
          {
            break;
          }
      }
    }
    if (~(data & ndata) && ~(address & naddress)) {
      Serial.print("NBVLC RX: Addr:");
      Serial.print(address);
      Serial.print(", Data:");
      Serial.println(data);
    }
    state = 0;
  }
  //---------------------------------------------------------
  if (test_delay_ctr++ > 50) {
    test_delay_ctr = 0;
    Serial.print("NBVLC TX: Addr:0, Data:");
    Serial.println(c);
    if (_start_send == 0) send_NEC(c);
  }
  //---------------------------------------------------------
}

// *************************************************************************
void test_eink_and_sensor_nbvlc() {
  if (state == 3) {
    int j;
    int address = 0x0000, naddress = 0x0000;
    int data = 0x0000, ndata = 0x0000;
    for (j = 0; j < 64; j = j + 2) {
      //Serial.print(ls[j]); Serial.print(","); Serial.print(ts[j]); Serial.print("|");
      //Serial.print(abs(ts[j]-ts[j+1])<10?0:1);
      switch (j >> 4) {
        case 0:
          {
            address |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 1:
          {
            naddress |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 2:
          {
            data |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 3:
          {
            ndata |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            //Serial.print((abs(ts[j]-ts[j+1])<10?0:1));
            break;
          }
        default:
          {
            break;
          }
      }
    }
    if (~(data & ndata) && ~(address & naddress)) {
      Serial.print("NBVLC RX: Addr:");
      Serial.print(address);
      Serial.print(", Data:");
      Serial.println(data);      
    }
    nbvlc_off();  // ************nbvlc_off() added here for low-power mode
    state = 0;
  }

  if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
    test_delay_ctr = 0;
    nbvlc_off();
    bme_print();
    //delay(2000); //****************************************************************************************
    epd_message();
    nbvlc_on();
    Serial.print("NBVLC TX: Addr:0, Data:");
    Serial.println(0x11);
    if (_start_send == 0) send_NEC(0x11);
  }
}

// *************************************************************************
void test_eink_and_sensor_nbvlc_normal() {
  if (state == 3) {
    int j;
    int address = 0x0000, naddress = 0x0000;
    int data = 0x0000, ndata = 0x0000;
    for (j = 0; j < 64; j = j + 2) {
      //Serial.print(ls[j]); Serial.print(","); Serial.print(ts[j]); Serial.print("|");
      //Serial.print(abs(ts[j]-ts[j+1])<10?0:1);
      switch (j >> 4) {
        case 0:
          {
            address |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 1:
          {
            naddress |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 2:
          {
            data |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 3:
          {
            ndata |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            //Serial.print((abs(ts[j]-ts[j+1])<10?0:1));
            break;
          }
        default:
          {
            break;
          }
      }
    }
    if (~(data & ndata) && ~(address & naddress)) {
      Serial.print("NBVLC RX: Addr:");
      Serial.print(address);
      Serial.print(", Data:");
      Serial.println(data);
    }
    state = 0;
  }

  if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
    test_delay_ctr = 0;
    bme_print();
    epd_message();
    nbvlc_on();
    Serial.print("NBVLC TX: Addr:0, Data:");
    Serial.println(0x11);
    if (_start_send == 0) send_NEC(0x11);
  }
}
// *************************************************************************
void test_sensor_nbvlc_normal() {
  if (state == 3) {
    int j;
    int address = 0x0000, naddress = 0x0000;
    int data = 0x0000, ndata = 0x0000;
    for (j = 0; j < 64; j = j + 2) {
      //Serial.print(ls[j]); Serial.print(","); Serial.print(ts[j]); Serial.print("|");
      //Serial.print(abs(ts[j]-ts[j+1])<10?0:1);
      switch (j >> 4) {
        case 0:
          {
            address |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 1:
          {
            naddress |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 2:
          {
            data |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 3:
          {
            ndata |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            //Serial.print((abs(ts[j]-ts[j+1])<10?0:1));
            break;
          }
        default:
          {
            break;
          }
      }
    }
    if (~(data & ndata) && ~(address & naddress)) {
      Serial.print("NBVLC RX: Addr:");
      Serial.print(address);
      Serial.print(", Data:");
      Serial.println(data);
    }
    state = 0;
  }

  if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
    test_delay_ctr = 0;
    bme_print();
    //epd_message();
    nbvlc_on();
    Serial.print("NBVLC TX: Addr:0, Data:");
    Serial.println(0x11);
    if (_start_send == 0) send_NEC(0x11);
  }
}
// *************************************************************************
void test_sensor_nbvlc_lowpower() {
  if (state == 3) {
    int j;
    int address = 0x0000, naddress = 0x0000;
    int data = 0x0000, ndata = 0x0000;
    for (j = 0; j < 64; j = j + 2) {
      //Serial.print(ls[j]); Serial.print(","); Serial.print(ts[j]); Serial.print("|");
      //Serial.print(abs(ts[j]-ts[j+1])<10?0:1);
      switch (j >> 4) {
        case 0:
          {
            address |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 1:
          {
            naddress |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 2:
          {
            data |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            break;
          }
        case 3:
          {
            ndata |= (abs(ts[j] - ts[j + 1]) < 10 ? 0 : 1) << ((j >> 1) & 0x0007);
            //Serial.print((abs(ts[j]-ts[j+1])<10?0:1));
            break;
          }
        default:
          {
            break;
          }
      }
    }
    if (~(data & ndata) && ~(address & naddress)) {
      Serial.print("NBVLC RX: Addr:");
      Serial.print(address);
      Serial.print(", Data:");
      Serial.println(data);      
    }
    nbvlc_off();  // ************nbvlc_off() added here for low-power mode
    state = 0;
  }

  if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
    test_delay_ctr = 0;
    nbvlc_off();
    bme_print();
    // epd_message();
    nbvlc_on();
    Serial.print("NBVLC TX: Addr:0, Data:");
    Serial.println(0x11);
    if (_start_send == 0) send_NEC(0x11);
  }
}
// *************************************************************************
void test_sensor_lowpower() {
    if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
    test_delay_ctr = 0;
    nbvlc_off();
    bme_print();
    // epd_message();
  }
}

//-----------------------------------------------------------------------------

void bme_setup() {
  Serial.println(F("BME680 test"));

  if (!bme.begin()) {
    Serial.println("Could not find a valid BME680 sensor, check wiring!");
    //while (1);
  }

  // Set up oversampling and filter initialization
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150);  // 320*C for 150 ms
}

void bme_print() {
  if (!bme.performReading()) {
    Serial.println("Failed to perform reading :(");
    return;
  }
  Serial.println();

  Serial.print("Temperature = ");
  Serial.print(bme.temperature);
  Serial.println(" *C");

  Serial.print("Pressure = ");
  Serial.print(bme.pressure / 100.0);
  Serial.println(" hPa");

  Serial.print("Humidity = ");
  Serial.print(bme.humidity);
  Serial.println(" %");

  Serial.print("Gas = ");
  Serial.print(bme.gas_resistance / 1000.0);
  Serial.println(" KOhms");

  Serial.print("Approx. Altitude = ");
  Serial.print(bme.readAltitude(SEALEVELPRESSURE_HPA));
  Serial.println(" m");

  Serial.println();
}

// void as_setup() {
//   Serial.println("test1: Checking AS3933 functionality");
//   if (!asTag.begin(38000)) {
//     Serial.println("Communication with AS3933 fails.");
//     return;
//   }

//   if (!asTag.doRcOscSelfCalib()) {
//     Serial.println("RC-oscillator not correctly calibrated.");
//     return;
//   }
//   Serial.println("RC-oscillator OK");

//   if (!asTag.setNrOfActiveAntennas(1) || !asTag.setListeningMode(As3933::LM_STANDARD) || !asTag.setFrequencyDetectionTolerance(As3933::FDT_BIG) || !asTag.setAgc(As3933::AGC_UP_DOWN, As3933::GR_NONE) || !asTag.setAntennaDamper(As3933::DR_NONE)) {
//     return;
//   }
//   if (!asTag.setWakeUpProtocol(As3933::WK_FREQ_DET_ONLY)) {
//     return;
//   }
//   delay(1000);
//   asTag.clear_wake();
//   // delay(1000);
//   Serial.println("Everything set up.");
// }


void as_setup()
{
    if(!asTag.begin(38000))
    {
        return;
    }

        if(!asTag.doRcOscSelfCalib())
    {
        return;
    }

    if(!asTag.setNrOfActiveAntennas(1) ||
        !asTag.setListeningMode(As3933::LM_STANDARD) ||
        !asTag.setFrequencyDetectionTolerance(As3933::FDT_BIG) ||
        !asTag.setAgc(As3933::AGC_UP_DOWN, As3933::GR_NONE) ||
        !asTag.setAntennaDamper(As3933::DR_NONE) || !asTag.setArtificialWakeUpTime(5))
    {
        return;
    }
    /*
    if(!asTag.setWakeUpProtocol(As3933::WK_FREQ_DET_ONLY))
    {
        return;
    }
    delay(1000);
    */
    asTag.clear_wake();
}





void nbvlc_on() {
  //set 1 bit, reprsent sending
  collected_data.running_mode |= (1 << 1); 
  send_periodic_frame_example(true);
  send_periodic_frame_example(true);
  ITimer0.setInterval(TIMER0_INTERVAL_MS / 10, TimerHandler0);
}

void nbvlc_off() {
  ITimer0.setInterval(TIMER0_INTERVAL_MS * 10, TimerHandler0);
}

// void nbvlc_off() {
//     ITimer0.setInterval(TIMER0_INTERVAL_MS * 10, TimerHandler0); // slight decrease by 0.05 mA
//     // CLear bit 1 
//     collected_data.running_mode &= ~(1 << 1);
//     send_periodic_frame_example(true);
// }

//-----------------------------------------------------------------------------
Paint paint(image, epd.bufwidth * 8, epd.bufheight);  //width should be the multiple of 8
float nums = 23.12;
void epd_message() {
  Serial.println("epd FULL");
  epd.Init(FULL);
  //epd.Display(IMAGE_DATA);    // *******************************COMMENTED ***************************

  //delay(2000);

  char str_buf[40];
  //float nums = 23.12;
  nums = bme.temperature;
  sprintf(str_buf, "T:%d.%02d deg. C", (int)nums, (int)(nums * 100 - 100 * (int)nums));
  paint.Clear(UNCOLORED);
  paint.DrawStringAt(8, 2, str_buf, &Font12, COLORED);

  nums = bme.pressure / 100.0;
  sprintf(str_buf, "P:%d.%02d HPa", (int)nums, (int)(nums * 100 - 100 * (int)nums));
  paint.DrawStringAt(8, 20, str_buf, &Font12, COLORED);

  nums = bme.humidity;
  sprintf(str_buf, "RH:%d.%02d %", (int)nums, (int)(nums * 100 - 100 * (int)nums));
  paint.DrawStringAt(8, 38, str_buf, &Font12, COLORED);

  nums = bme.gas_resistance / 1000.0;
  sprintf(str_buf, "Gas:%d.%02d kOhm", (int)nums, (int)(nums * 100 - 100 * (int)nums));
  paint.DrawStringAt(8, 56, str_buf, &Font12, COLORED);

  //  paint.DrawStringAt(8, 74, "Hello world", &Font12, COLORED);
  //  paint.DrawStringAt(8, 92, "Hello world", &Font12, COLORED);
  epd.Display(image);  //1
  //epd.Display1(image);//1
  //epd.Display1(image);//1
  //epd.Display1(image);//1
  // delay(2000);
  /*
  paint.Clear(UNCOLORED);
  paint.DrawRectangle(2,2,50,50,COLORED);
  paint.DrawLine(2,2,50,50,COLORED);
  paint.DrawLine(2,50,50,2,COLORED);
  paint.DrawFilledRectangle(52,2,100,50,COLORED);
  paint.DrawLine(52,2,100,50,UNCOLORED);
  paint.DrawLine(100,2,52,50,UNCOLORED);
  epd.Display1(image);//2
  
  paint.Clear(UNCOLORED);
  paint.DrawCircle(25,25,20,COLORED);
  paint.DrawFilledCircle(75,25,20,COLORED);
  epd.Display1(image);//3
  
  paint.Clear(UNCOLORED);
  epd.Display1(image);//4

  delay(2000);
*/
  /*
  Serial.println("epd PART");
  epd.DisplayPartBaseImage(IMAGE_DATA);
  char i = 0;
  for (i = 0; i < 10; i++) {
    Serial.println("e-Paper PART IMAGE_DATA");
    epd.Init(PART);
    epd.DisplayPart(IMAGE_DATA);
    Serial.println("e-Paper PART Clear");
	  epd.Init(PART);
    epd.ClearPart();
    delay(2000);
  }
  */

  //**************** BETTER TO COMMENT ELSE CURRENT DOUBLES AFTER 2 seconds
  // epd.Init(FULL);
  // Serial.println("e-Paper clear and sleep");
  // //epd.Clear();
  // epd.Sleep();
  //*********************************
}

void epd_image() {
  Serial.println("epd FULL");
  epd.Init(FULL);
  int i;
  for (i = 0; i < 16 * (120 + 120 + 10) / 2; i++) image[i] = 0xff;
  epd.Display(IMAGE_DATA);

  delay(2000);
  //epd.Display(image);//1
  //epd.Display1(image);//1
  //epd.Display1(image);//1
  //epd.Display1(image);//1
  //delay(2000);
  //epd.Init(FULL);
  Serial.println("e-Paper clear and sleep");
  //epd.Clear();
  epd.Sleep();
}

// *************************************************************************
void test_eink_and_sensor_nbvlc_TX_readings() {
nbvlc_off();  // ************nbvlc_off() added here for low-power mode
state = 0;

if (test_delay_ctr++ > 50) {  // 50 iterations * 100 milliseconds/iteration
test_delay_ctr = 0;
nbvlc_off();
bme_print();
epd_message();

// Data (temperature, pressure, humidity, gas_resistance, altitude)
float data[] = {
bme.temperature,
(float)bme.pressure/ 100.0,
bme.humidity,
(float)bme.gas_resistance/ 1000.0,
bme.readAltitude(SEALEVELPRESSURE_HPA)
};  

size_t dataSize = sizeof(data) / sizeof(data[0]);

uint8_t frame[64]; // Adjust size as needed
size_t index = 0;
// Start of frame
frame[index++] = START_FRAME;
// Address (2 bytes)
frame[index++] = ADDRESS_Source;
frame[index++] = ADDRESS_Destination;

// Command
frame[index++] = CMD_SEND;

// Number of data bytes
frame[index++] = dataSize * sizeof(float);

// Copy actual data (convert float to bytes)
for (size_t i = 0; i < dataSize; i++) {
uint8_t* dataBytes = (uint8_t*)&data[i];
for (size_t j = 0; j < sizeof(float); j++) {
  frame[index++] = dataBytes[j];
}
}
// Checksum
frame[index++] = calculateChecksum(&frame[0], index - 1);

// End of frame
frame[index++] = END_FRAME;

// Send the frame using send_NEC
Serial.print("Frame: ");
//nbvlc_on();
for (size_t i = 0; i < index; i++) {
// while (_start_send) {
// // Wait until the previous byte is sent
// delay(300); // Small delay to allow for sending time
// }

if (_start_send == 0) {
  nbvlc_on();
  send_NEC(frame[i]);
  delay(100);
  nbvlc_off();
} 
delay(200);
// Print the frame byte in hexadecimal
Serial.print(frame[i], HEX);
Serial.print(" ");
}

Serial.println(" ");
Serial.println("Send successfully");
}
}
//**************************************************************************************************************
void test_eink_and_sensor_nbvlc_TX_readings_2() {   // VLC FRAME

if (test_delay_ctr++ > 100) {  // 50 iterations * 100 milliseconds/iteration
test_delay_ctr = 0;

 nbvlc_off();
 bme_print();
float temperature = bme.temperature;
float pressure = bme.pressure / 100.0;
float humidity = bme.humidity;
float gas = bme.gas_resistance / 1000.0;
 epd_message();

// Prepare sensor and location data
SensorData sensorData = prepareSensorData(temperature, pressure, humidity, gas);

// Define current location coordinates
float x_coord = 9.23;  // X coordinate in meters
float y_coord = -4.56; // Y coordinate in meters
float z_coord = 9.89;  // Z coordinate in meters

LocationData locationData = prepareLocationData(x_coord, y_coord, z_coord);

// Create and send frame
Frame frame;
frame.startMarker  = START_MARKER;
frame.sourceAddr   = MY_ADDRESS;
frame.destAddr     = 0x01;
frame.command      = CMD_COMBINED;  // Change this to any command type you need

uint8_t* payloadPtr = frame.payload;
// Automatically set payload size and copy data based on command type
switch(frame.command) {
    case CMD_SENSOR:
        frame.payloadSize = sizeof(SensorData);
        memcpy(payloadPtr, &sensorData, sizeof(SensorData));
        break;
    case CMD_LOCATION:
        frame.payloadSize = sizeof(LocationData);
        memcpy(payloadPtr, &locationData, sizeof(LocationData));
        break;
    case CMD_COMBINED:
        frame.payloadSize = sizeof(SensorData) + sizeof(LocationData);
        memcpy(payloadPtr, &sensorData, sizeof(SensorData));
        payloadPtr += sizeof(SensorData);
        memcpy(payloadPtr, &locationData, sizeof(LocationData));
        break;
    case CMD_REQ_SENSOR:
    case CMD_REQ_LOC:
    case CMD_REQ_BOTH:
    case CMD_ACK:
    case CMD_NACK:
    default:
        frame.payloadSize = 0;  // No payload for request/ack/nack commands
        break;
}

frame.checksum    = calculateChecksum(&frame);
frame.endMarker   = END_MARKER;
 nbvlc_on();
// Transmit the frame
transmitFrame(frame);

// Debug output - only show if we're sending sensor data
if (frame.command == CMD_SENSOR) {
    Serial.print("T:"); Serial.print(temperature, 1);
    Serial.print("°C P:"); Serial.print(pressure, 1);
    Serial.print("hPa H:"); Serial.print(humidity, 1);
    Serial.print("% G:"); Serial.print(gas, 2);
    Serial.println("kΩ");
}

if ( frame.command == CMD_COMBINED) {
    Serial.print("T:"); Serial.print(temperature, 1);
    Serial.print("°C P:"); Serial.print(pressure, 1);
    Serial.print("hPa H:"); Serial.print(humidity, 1);
    Serial.print("% G:"); Serial.print(gas, 2);
    Serial.print("kΩ X:");Serial.print(x_coord,2);
    Serial.print("m Y:");Serial.print(y_coord,2);
    Serial.print("m Z:");Serial.print(z_coord,2);
    Serial.println("m");
}
if ( frame.command == CMD_LOCATION) {
    Serial.print("X:");Serial.print(x_coord,2);
    Serial.print("m Y:");Serial.print(y_coord,2);
    Serial.print("m Z:");Serial.print(z_coord,2);
    Serial.println("m");
}
}
}



// Callback function to handle received data
void bleRxCallback(uint16_t conn_handle) {
  (void) conn_handle;
  while (bleuart.available()) {
    char c = (char)bleuart.read();
    Serial.print(c); // Print received data to Serial Monitor
  }
}



