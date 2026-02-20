#ifndef NBVLC_H
#define NBVLC_H



// 1. Tell every TU about the handler signature:
// #ifdef __cplusplus
// extern "C" {
// #endif
// void TimerHandler0(void);
// #ifdef __cplusplus
// }
// #endif

// extern NRF52Timer ITimer0;

void nbvlc_setup();
void updateDC(uint16_t level);
void send_NEC(int8_t c);
int nbvlc_rx_run();
int nbvlc_send(unsigned char c);
unsigned char nbvlc_get_data();
unsigned char nbvlc_get_address();
void nbvlc_lamp_off();
void nbvlc_lamp_on(float p);
void nbvlc_lamp_p(float p);
void test_nbvlc_frame(uint8_t* send_data, int count);
void sendFrame(uint8_t* send_data, int count);
void processFrame();
void setInterval_time0(uint8_t input);
void nbvlc_on();
void nbvlc_off();

//************** JUNAID ADDED VLC FRAMES ******************
#define MAX_PAYLOAD_SIZE 16         // Maximum payload size in bytes
// Command types for different data types
#define CMD_REQ_SENSOR  0x06        // Request sensor data
#define CMD_REQ_LOC     0x07        // Request location data
#define CMD_REQ_BOTH    0x08        // Request both sensor and location data
#define MY_ADDRESS      0xF1       // Add this with other #define statements
#define START_MARKER    0xAA        // Frame start marker (binary: 10101010)
#define END_MARKER      0x55        // Frame end marker (binary: 01010101)

 // Frame Structure (23 bytes total)
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
void processReceivedData();
void transmitFrame(const Frame& frame);
uint8_t calculateChecksumFrame(const Frame* frame);


extern unsigned long vlc_lastResetTime;
extern void vlc_reportAndResetBytesCount();
extern unsigned long vlc_lastResetTime_RX;
extern void vlc_reportAndResetBytesCount_RX();
//**********************************************************

#ifndef SHARED_H
#define SHARED_H
// Define the frame buffer size
#define FRAME_BUFFER_SIZE 64
// Declare the shared volatile variable
extern volatile bool frameFlag;
extern size_t frameIndex;
extern size_t frameIndex_clone;
// Declare the shared frame buffer
extern uint8_t frameBuffer[FRAME_BUFFER_SIZE];
extern const uint8_t ADDRESS_Source; // Source address

extern const uint8_t ADDRESS_Destination; // Destination address (Node). Range is 00-F0
extern const uint8_t CMD_SEND;   // Example command to send data
extern const uint8_t CMD_RECEIVE; // Example command to receive data

extern const uint8_t START_FRAME; // Start of Text (STX)
extern const uint8_t END_FRAME;   // End of Text (ETX)


#endif



#define TIMER_INTERRUPT_DEBUG         0
#define _TIMERINTERRUPT_LOGLEVEL_     3


#define USING_TIMER   false   //true
#define TIMER0_INTERVAL_MS        500   //1000

#define pinToUse   1 //TX PIN - 1 //1 xiao core 23
#define IN1 2        //RX pin - 0 xiao core 11

#endif // NBVLC_H