#include "Arduino.h"
#include "nbvlc.h"
#include "NRF52TimerInterrupt.h"
#include "nRF52_PWM.h"
#include <bluefruit.h>
#include "bles.h"
#include "serial_frame_util.h"

nRF52_PWM* PWM_Instance;
NRF52Timer ITimer0(NRF_TIMER_4);

/**********Senhui Add**********/
#define FRAME_BUFFER_SIZE 64
uint8_t frameBuffer[FRAME_BUFFER_SIZE]; // Buffer to store received frame
volatile bool frameReceived = false;           // Flag indicating a complete frame received
// Define start and end frame markers
const uint8_t START_FRAME = 0xAA; // Start of Text (STX)
const uint8_t END_FRAME = 0xEE;   // End of Text (ETX)
// Addresses and commands for protocol
const uint8_t ADDRESS_Source = 0xF1; // Source address (Gateway). Range is F1-FF
const uint8_t ADDRESS_Destination = 0x01; // Destination address (Node). Range is 00-F0
const uint8_t CMD_SEND = 0x01;   // Example command to send data
const uint8_t CMD_RECEIVE = 0x02; // Example command to receive data

size_t frameIndex = 0;              // Index for the current frame byte
size_t frameIndex_clone = 0;              // Index for the current frame byte
volatile bool frameFlag = false;    // Flag indicating a complete frame received

int receivedData = 0;  // The byte currently being received
volatile uint8_t bitIndex = 0;      // Tracks the bit position in the byte
volatile bool dataReady = false;    // Flag indicating a byte is fully received
uint32_t t0 = 0, dt = 0;    // Timing variables
/**********Senhui Add**********/
volatile uint8_t frameQueue[FRAME_BUFFER_SIZE];
volatile size_t frameQueueLen = 0;
volatile size_t frameSendIndex = 0;
volatile bool frameSending = false;

#define TIMER0_INTERVAL_MS 500

//**************************************** JUNAID VLC FRAME DEFINITIONS ***********************************************

// Command types for different data types
#define CMD_SENSOR      0x01        // Sending only sensor data
#define CMD_LOCATION    0x02        // Sending only location data
#define CMD_COMBINED    0x03        // Sending both sensor and location data
#define CMD_ACK         0x04        // Acknowledgment message
#define CMD_NACK        0x05        // Negative acknowledgment
#define CMD_REQ_SENSOR  0x06        // Request sensor data
#define CMD_REQ_LOC     0x07        // Request location data
#define CMD_REQ_BOTH    0x08        // Request both sensor and location data

// Frame size calculations
#define FRAME_SIZE_BYTES  23        // Total frame size: header(4) + payloadInfo(17) + footer(2)
#define FRAME_SIZE_BITS   (FRAME_SIZE_BYTES * 8)  // 184 bits total
#define LAST_CHUNK_BITS   ((FRAME_SIZE_BITS % 32) ? (FRAME_SIZE_BITS % 32) : 32)  // 24 bits in last chunk
#define FRAME_CHUNKS    6           // Number of 32-bit chunks needed: ceil(184/32) = 6

// Transmission timing constants
#define CHUNK_DELAY_MS    100        // Delay between chunk transmissions
#define FRAME_DELAY_MS    5000      // Delay between complete frames

// Frame reception buffer
uint32_t receivedDataChunks[FRAME_CHUNKS];  // Buffer for received chunks
int receivedChunks = 0;               // Count of received chunks
bool isReceiving = false;             // Frame reception in progress flag

// Frame chunk layout in 32-bit chunks
/*
 * Chunk 0: [startMarker][sourceAddr][destAddr][command]     - Frame header
 * Chunk 1: [payloadSize][payload[0:2]]                     - Start of payload
 * Chunk 2: [payload[3:6]]                                  - Payload data
 * Chunk 3: [payload[7:10]]                                 - Payload data
 * Chunk 4: [payload[11:14]]                                - Payload data
 * Chunk 5: [payload[15]][checksum][endMarker][padding]     - End of payload and frame footer
 */

struct __attribute__((packed)) LocationData {
    int16_t x;             // X coordinate * 100 (-1000 to 1000 for -10.00 to 10.00m)
    int16_t y;             // Y coordinate * 100 (-1000 to 1000 for -10.00 to 10.00m)
    int16_t z;             // Z coordinate * 100 (-1000 to 1000 for -10.00 to 10.00m)
};
struct __attribute__((packed)) SensorData {
    int16_t temperature;    // Temperature in °C * 10 (-400 to 850 for -40.0 to 85.0°C)
    uint16_t pressure;      // Pressure in hPa * 10 - 300 (3000 to 11000 for 300.0 to 1100.0 hPa)
    uint16_t humidity;      // Humidity in % * 10 (0 to 1000 for 0.0 to 100.0%)
    uint16_t gas;          // Gas resistance in kΩ * 100 (0 to 20000 for 0.0 to 200.0 kΩ)
};
// Function to calculate checksum for received frame
uint8_t calculateChecksumFrame(const Frame* frame) {
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
    if(
      frame->command != CMD_SENSOR && 
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
    switch(frame->command) 
    {
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
                Serial.println("Error: Request should have no payload");
                return false;
            }
            break;
    }

    // Verify checksum
    uint8_t calculatedChecksumF = calculateChecksumFrame(frame);
    if(calculatedChecksumF != frame->checksum) {
        Serial.println("Error: Checksum mismatch");
        return false;
    }
    return true;
}



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

//****************************************
float frequency = 38000.0f;
float dutyCycle = 0.0f;
uint32_t time_t_now = 0;
uint32_t time_t_old = 0;
uint32_t _c = 0; //0x00ff107f;
int _start_send = 0;
int _bit_i = 0;
volatile uint64_t Timer0Count = 0;
float NNF = 1.28; //1.28;

int t1 = Timer0Count;
int state = 0;
int long ls[64],ts[64];
int i = 0;
int long len = 0;

int address = 0x0000, naddress = 0x0000;
int data = 0x0000, ndata = 0x0000;

float lamp_pwm = 20.0f;


//********************************* JUNAID VLC FRAME TX  *****************************

//uint32_t _c  = 0;                   // Current chunk being transmitted

// Function to send 32 bits of frame data
void send_32bits(uint32_t chunk)
{
    _c = chunk;  // Store chunk in global variable for state machine
    _start_send = 1;  // Start transmission state machine
    while(_start_send != 0) {  // Wait for transmission to complete
        delay(1);
    }
}



//********************************* JUNAID VLC FRAME DECODING *****************************
void processReceivedData() {
    if(state == 3) {
        int j;
        uint32_t chunk = 0;
        
        // Process bits from timing data
        int bitsToProcess = (receivedChunks == FRAME_CHUNKS - 1) ? LAST_CHUNK_BITS : 32;
        
        // Process only the required number of bits
        for(j = 0; j < bitsToProcess * 2; j += 2) 
        {
            chunk |= ((uint32_t)(abs(ts[j]-ts[j+1]) < 10 ? 0:1)) << (j/2);
        }

        // Check if this is the start of a new frame
        if(!isReceiving) {
            uint8_t* firstByte = (uint8_t*)&chunk;
            if(*firstByte == START_MARKER) {
                isReceiving = true;
                receivedChunks = 0;
                memset(receivedDataChunks, 0, sizeof(receivedDataChunks));  // Clear buffer
            }
        }

        if(isReceiving) {
            if(receivedChunks < FRAME_CHUNKS) {
                // For the last chunk, only store the valid bits
                if(receivedChunks == FRAME_CHUNKS - 1) {
                    // Clear any potential garbage in unused bits
                    uint32_t mask = (1UL << LAST_CHUNK_BITS) - 1;
                    chunk &= mask;
                }
                
                receivedDataChunks[receivedChunks++] = chunk;
                
                // Check if we have received all chunks
                if(receivedChunks >= FRAME_CHUNKS) {
                    Frame* receivedFrame = (Frame*)receivedDataChunks;
                    
                    // Validate the frame
                    if(validateFrame(receivedFrame)) {
                        // Process based on command type
                        switch(receivedFrame->command) {
                            case CMD_SENSOR: {
                                SensorData* sensorData = (SensorData*)receivedFrame->payload;
                                
                                // Validate sensor data ranges
                                float temperature = sensorData->temperature / 10.0f;
                                float pressure = (sensorData->pressure / 10.0f) + 300.0f;
                                float humidity = sensorData->humidity / 10.0f;
                                float gas = sensorData->gas / 100.0f;

                                if(temperature >= -40.0f && temperature <= 85.0f &&
                                   pressure >= 300.0f && pressure <= 1100.0f &&
                                   humidity >= 0.0f && humidity <= 100.0f &&
                                   gas >= 0.0f && gas <= 200.0f) {
                                    
                                    Serial.print("Valid Sensor Data - T:");
                                    Serial.print(temperature, 1);
                                    Serial.print("°C P:");
                                    Serial.print(pressure, 1);
                                    Serial.print("hPa H:");
                                    Serial.print(humidity, 1);
                                    Serial.print("% G:");
                                    Serial.print(gas, 2);
                                    Serial.println("kΩ");
                                } else {
                                    Serial.println("Error: Sensor data out of range");
                                }
                                break;
                            }
                            
                            case CMD_LOCATION: {
                                LocationData* locData = (LocationData*)receivedFrame->payload;
                                
                                float x = locData->x / 100.0f;
                                float y = locData->y / 100.0f;
                                float z = locData->z / 100.0f;

                                if(x >= -10.0f && x <= 10.0f &&
                                   y >= -10.0f && y <= 10.0f &&
                                   z >= -10.0f && z <= 10.0f) {
                                    
                                    Serial.print("Valid Location - X:");
                                    Serial.print(x, 2);
                                    Serial.print("m Y:");
                                    Serial.print(y, 2);
                                    Serial.print("m Z:");
                                    Serial.print(z, 2);
                                    Serial.println("m");
                                } else {
                                    Serial.println("Error: Location data out of range");
                                }
                                break;
                            }
                            
                            case CMD_COMBINED: {
                                SensorData* sensorData = (SensorData*)receivedFrame->payload;
                                LocationData* locData = (LocationData*)(receivedFrame->payload + sizeof(SensorData));
                                
                                float temperature = sensorData->temperature / 10.0f;
                                float pressure = (sensorData->pressure / 10.0f) + 300.0f;
                                float humidity = sensorData->humidity / 10.0f;
                                float gas = sensorData->gas / 100.0f;
                                float x = locData->x / 100.0f;
                                float y = locData->y / 100.0f;
                                float z = locData->z / 100.0f;

                                if(temperature >= -40.0f && temperature <= 85.0f &&
                                   pressure >= 300.0f && pressure <= 1100.0f &&
                                   humidity >= 0.0f && humidity <= 100.0f &&
                                   gas >= 0.0f && gas <= 200.0f &&
                                   x >= -10.0f && x <= 10.0f &&
                                   y >= -10.0f && y <= 10.0f &&
                                   z >= -10.0f && z <= 10.0f) {
                                    
                                    Serial.print("Valid Combined Data - T:");
                                    Serial.print(temperature, 1);
                                    Serial.print("°C P:");
                                    Serial.print(pressure, 1);
                                    Serial.print("hPa H:");
                                    Serial.print(humidity, 1);
                                    Serial.print("% G:");
                                    Serial.print(gas, 2);
                                    Serial.print("kΩ | X:");
                                    Serial.print(x, 2);
                                    Serial.print("m Y:");
                                    Serial.print(y, 2);
                                    Serial.print("m Z:");
                                    Serial.print(z, 2);
                                    Serial.println("m");
                                } else {
                                    Serial.println("Error: Data out of valid ranges");
                                }
                                break;
                            }

                        }
                    } else {
                        Serial.println("Frame validation failed");
                    }
                    isReceiving = false;
                    receivedChunks = 0;
                }
            } else {
                Serial.println("Error: Too many chunks received");
                isReceiving = false;
                receivedChunks = 0;
            }
        }        
        state = 0;  // Reset for next chunk
    }
}

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

// ****************************************************************************************



// void TimerHandler0()
// {
//   static bool toggle0 = false;

//   // Flag for checking to be sure ISR is working as SErial.print is not OK here in ISR
//   Timer0Count++;

//   //timer interrupt toggles pin LED_BUILTIN
//   //digitalWrite(LED_BUILTIN, toggle0);
//   toggle0 = !toggle0;
  
//   switch(_start_send)
//   {
//     case 1: {
//       updateDC(50);
//       time_t_now = 0;
//       _start_send = 2;
//       break;
//     }
//     case 2: {
//       time_t_now++;
//       if(time_t_now >= 131*NNF) _start_send = 3;
//       break;
//     }
//     case 3: {
//       updateDC(0);
//       time_t_now = 0;
//       _start_send = 4;
//       break;
//     }
//     case 4: {
//       time_t_now++;
//       if(time_t_now >= 65*NNF) 
//       {
//         _start_send = 5;
//         _bit_i = 0;
//       } 
//       break;
//     }
//     case 5: {
//       updateDC(50);
//       time_t_now = 0;
//       _start_send = 6;
//       break;
//     }
//     case 6: {
//       time_t_now++;
//       if(time_t_now >= 7*NNF) _start_send = 7;
//       break;
//     }        
//     case 7: {
//       updateDC(0);
//       time_t_now = 0;
//       if(((0x00000001<<_bit_i)&(_c))) _start_send = 8; else _start_send = 9;
//       break;
//     }
//     case 8: {
//       time_t_now++;
//       if(time_t_now >= 21*NNF) _start_send = 10;
//       break;
//     }
//     case 9: {
//       time_t_now++;
//       if(time_t_now >= 7*NNF) _start_send = 10;
//       break;
//     }  
//     case 10: {
//       _bit_i = _bit_i + 1;
//       if(_bit_i >= 32) _start_send = 11; else _start_send = 5;
//       break;
//     }
//     case 11: {
//       updateDC(50);
//       time_t_now = 0;
//       _start_send = 12;
//       break;
//     }
//     case 12: {
//       time_t_now++;
//       if(time_t_now >= 7*NNF) _start_send = 13;
//       break;
//     }        
//     case 13: {
//       updateDC(0);
//       _start_send = 0;
//       break;
//     }

//     default: {
//       break;
//     }
//   }
  
// }

void encodeAndStartSend(uint8_t c) {
    _c = ((uint32_t)c << 16) | ((~(uint32_t)c) << 24) | 0x0000FF00;
    _start_send = 1;

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
}


#define TIMER_INTERVAL_US       (TIMER0_INTERVAL_MS * 100)  // 500ms / 10 = 50us
#define TIMER_INTERVAL_MS       (TIMER_INTERVAL_US / 1000.0)  // 0.05ms
#define DESIRED_DELAY_MS        300
// #define DELAY_BETWEEN_BYTES     ((int)(DESIRED_DELAY_MS / TIMER_INTERVAL_MS))  // = 6000
#define DELAY_BETWEEN_BYTES     6000

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


void setInterval_time0(uint8_t input)
{
  ITimer0.setInterval(int(input / 10), TimerHandler0);  
}

void nbvlc_on() {
  collected_data.running_mode |= (1 << 1); 
  send_periodic_frame_example(true);
  send_periodic_frame_example(true);
  ITimer0.setInterval(TIMER0_INTERVAL_MS / 10, TimerHandler0);
}

// void nbvlc_off() {
//   ITimer0.setInterval(TIMER0_INTERVAL_MS * 10, TimerHandler0);
// }

void nbvlc_off() {
    ITimer0.setInterval(TIMER0_INTERVAL_MS * 10, TimerHandler0); // slight decrease by 0.05 mA
}

void nbvlc_setup() 
{
// Interval in microsecs
  if (ITimer0.attachInterruptInterval(TIMER0_INTERVAL_MS / 10, TimerHandler0))
  {
    Serial.print(F("Starting ITimer0 OK, millis() = "));
    Serial.println(millis());
  }
  else
    Serial.println(F("Can't set ITimer0. Select another freq. or timer"));

  pinMode(pinToUse, OUTPUT);
  pinMode(IN1, INPUT_PULLUP);
  digitalWrite(pinToUse,LOW);

  attachInterrupt(digitalPinToInterrupt(IN1), in1_handler, CHANGE);
  PWM_Instance = new nRF52_PWM(pinToUse, frequency, dutyCycle);

}

void updateDC(uint16_t level)
{
  PWM_Instance->setPWM(pinToUse,(level?frequency:frequency*2),lamp_pwm);
}

void send_NEC(int8_t c)
{
  _c = ((uint32_t)c)<<16 | (~(uint32_t)c)<<24 | 0x0000ff00;
  //Serial.println(_c);s
  _start_send = 1;
}

int nbvlc_rx_run()
{
  if(state == 3) 
  {
    int j;
    address = 0x0000; naddress = 0x0000;
    data = 0x0000; ndata = 0x0000;

    for(j=0;j<64;j=j+2)
    {
      switch(j>>4)
      {
        case 0: {
          address |= (abs(ts[j]-ts[j+1])<10?0:1) << ((j>>1)&0x0007);
          break;
        }
        case 1: {
          naddress |= (abs(ts[j]-ts[j+1])<10?0:1) << ((j>>1)&0x0007);
          break;
        }        
        case 2: {
          data |= (abs(ts[j]-ts[j+1])<10?0:1) << ((j>>1)&0x0007);
          break;
        }    
        case 3: {
          ndata |= (abs(ts[j]-ts[j+1])<10?0:1) << ((j>>1)&0x0007);
          break;
        }              
        default: {
          break;
        }
      }
    }
    state = 0;
    if(~(data&ndata) && ~(address&naddress) )
    {
      int t1 = Timer0Count;
      return 1;
    }
  }
  return 0;
}

unsigned char nbvlc_get_data() {return data;}
unsigned char nbvlc_get_address() {return address;}

int nbvlc_send(unsigned char c)
{   
    if(_start_send == 0) {send_NEC(c); return 1;} else {return 0;}
}

void nbvlc_lamp_off()
{
  lamp_pwm = 0.0f;
  PWM_Instance->setPWM(pinToUse,frequency*2,lamp_pwm);
}

void nbvlc_lamp_on(float p)
{
  lamp_pwm = p;
  PWM_Instance->setPWM(pinToUse,frequency*2,lamp_pwm);
}

void nbvlc_lamp_p(float p)
{
  lamp_pwm = p;
  PWM_Instance->setPWM(pinToUse,frequency*2,lamp_pwm);
}


/************Senhui Add************/
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
    // if (collected_data.spare_3 == 0)
    // {
    //   vlc_lastResetTime_RX = millis();//start to record time
    // }
    collected_data.spare_3  = collected_data.spare_3  + 4;
    //set 7 bit. maybe impact vlc to receive the data?
    collected_data.running_mode |= (1 << 7); 
    send_periodic_frame_example(); //send node status
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
        Serial.print(receivedData, HEX);
        Serial.print(" ");

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
    send_periodic_frame_example(); //send node status
}


void swapBytes(uint8_t* bytes) {
    uint8_t temp = bytes[0];
    bytes[0] = bytes[3];
    bytes[3] = temp;
    temp = bytes[1];
    bytes[1] = bytes[2];
    bytes[2] = temp;
}

// float data[5] = {23.0, 1030.0,33.0, 27.0, 22.0};
// Function to send frame
void sendFrame(uint8_t* data, int count) {
    uint8_t frame[64]; // Adjust size as needed
    size_t index = 0;

    // Start of frame
    frame[index++] = START_FRAME;

    // Address (2 bytes)
    frame[index++] = ADDRESS_Source;
    // frame[index++] = ADDRESS_Destination;
    frame[index++] = data[0]; // ADDRESS_Destination

    // Command
    // frame[index++] = CMD_SEND;
    frame[index++] = data[1]; // Command
    
    if (data[1] == CMD_RECEIVE)
    {
      count = 0;// only require node send data to gateway
    }

    // Number of data bytes
    frame[index++] = count * sizeof(uint8_t);

    // Copy actual data (just copy the uint8_t data as is)
    for (int i = 0; i < count; i++) {
        frame[index++] = data[2+i];  // Copy each uint8_t byte
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

    Serial.print("Queuing frame: ");
    for (size_t i = 0; i < index; i++) {
        Serial.print(frame[i], HEX);
        Serial.print(" ");
    }
    Serial.println();
    //set 1 bit, reprsent sending
    collected_data.running_mode |= (1 << 1); 
    send_periodic_frame_example();
    queueFrameForSend(frame, index);
}






void test_nbvlc_frame(uint8_t* send_data, int count) {
Serial.println("Prepare frame");
sendFrame(send_data, count); // Send data
delay(100);
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
  send_periodic_frame_example(); //send node status
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
  send_periodic_frame_example(); //send node status
}

/************Senhui Add************/
