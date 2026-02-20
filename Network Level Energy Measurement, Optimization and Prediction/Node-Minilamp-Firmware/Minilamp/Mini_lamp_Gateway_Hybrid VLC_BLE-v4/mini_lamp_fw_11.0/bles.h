#include <bluefruit.h>



#define SCAN_TYPE_NULL 0
#define SCAN_TYPE_LIST 1
#define SCAN_TYPE_WRITE 2


// Define constants for your parameters if helpful
#define PHY_1M 1
#define PHY_2M 2
#define PHY_3M 3 // For Long Range
#define BLE_FRAME_BUFFER_SIZE 128

// Define a struct to hold the return values
typedef struct {
    uint8_t addr_ble_rec[6];
    uint8_t command;
    uint8_t length;
    size_t index;
} BLE_Data;

void bles_setup();
void scan_callback(ble_gap_evt_adv_report_t* report);
void connect_callback(uint16_t conn_handle);
void disconnect_callback(uint16_t conn_handle, uint8_t reason);
void bles_start_scan(int sec, int type = SCAN_TYPE_LIST, String * mac = NULL, uint8_t data = 0x00);
void bles_stop_scan();
bool bles_is_on();
int bles_written();
void sendBLEFrame(uint8_t* data, int count);
bool extractDataFromCmd(const String& cmd, uint8_t** send_data, int& data_len);
void CMD_BLE_Intervel_Process();
void CMD_PHY_Rate_Process();
void CMD_MTU_Process();
void reportAndResetBytesCount();
void reportAndResetBytesCount_RX();
BLE_Data processFramePayload();

extern volatile bool ble_frameFlag;    // Flag indicating a complete ble frame received
extern const uint8_t CMD_Running_State; //Power mode
extern const uint8_t CMD_Wakeup; //Wakeup 
extern const uint8_t CMD_Communcation_mode; //Communcation mode, choose vlc or ble
extern const uint8_t CMD_VLC_Protocol; //vlc bit, byte, compact level
extern const uint8_t CMD_VLC_Intervel; //Intervel
extern const uint8_t CMD_PWM; //PWM
extern const uint8_t CMD_VLC_Unit; //MTU
extern const uint8_t CMD_BLE_Protocol; //BLE_Protocol
extern const uint8_t CMD_BLE_Intervel; //BLE_Intervel
extern const uint8_t CMD_PHY_Rate; //PHY_Rate
extern const uint8_t CMD_MTU; //MTU
extern const uint8_t CMD_BLE_Unit; //Unit
extern const uint8_t CMD_BLE_OFF_ON; //BLE OFF/ON
extern const uint8_t CMD_BLE_FAST_TIME;
extern const uint8_t CMD_VLC_OFF_ON; //VLC OFF/ON
extern const uint8_t CMD_EINK_OFF_ON; //VLC OFF/ON
extern const uint8_t CMD_SENSING_OFF_ON; //VLC OFF/ON

extern const uint8_t CMD_VLC_SEND_Auto; //SEND VLC AUTO FOR TEST
extern const uint8_t CMD_BLE_SEND_Auto; //SEND BLE AUTO FOR TEST

extern unsigned long lastResetTime;
extern unsigned long lastResetTime_RX;

extern unsigned long last_ble_activity_time;
extern const unsigned long BLE_IDLE_TIMEOUT; // 10 seconds

extern uint16_t current_conn_handle;
extern uint8_t target_mac[6];
extern int scan_type;
extern int update_char;
extern bool BLE_command_received;
extern bool isConnected;
extern bool BLE_response_frame_received; 
extern uint8_t BLE_frameBuffer[BLE_FRAME_BUFFER_SIZE];
