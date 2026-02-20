#include <bluefruit.h>
#include "nbvlc.h"
#include "bles.h"
#include "serial_frame_util.h"
// Arduino Code for Hybrid VLC and BLE Communication Module a.k.a. Mini Gateway

#define BLE_WRITE_DURATION 10 //How many seconds to attempt update of BLE characteristic

// Serial baud rate
const long BAUD_RATE = 115200;

// Command Definitions
#define CMD_WAKEUP "#WUP"
#define CMD_SEND_VLC_CHAR "#VLS"
#define CMD_RECEIVE_VLC_CHAR "#RVC"
#define CMD_LISTEN_BLE_ADVERTISE "#LBA"
#define CMD_WRITE_BLE_CHAR "#WBL"
#define CMD_VLC_LAMP_OFF "#LOF"
#define CMD_VLC_LAMP_ON "#LON"
#define CMD_VLC_LAMP_P "#LAP"
#define CMD_HELP       "#HLP"
/**********Senhui Add**********/
#define CMD_SEND_VLC_FRAME   "#VLF" // send frame. e.g., #VLF Address_Destination, CMD, DATA
#define CMD_SEND_BLE_FRAME   "#BLF" // send frame. e.g., #VBF Address_Destination, CMD, DATA

/**********Senhui Add**********/
// #define FRAME_BUFFER_SIZE 64




// volatile uint8_t frameQueue[FRAME_BUFFER_SIZE];
// volatile size_t frameQueueLen = 0;
// volatile size_t frameSendIndex = 0;
// volatile bool frameSending = false;

CollectedData_t collected_data;






#define MAX_TOKENS  16   // adjust up if you need more data bytes

// #define MAX_FRAME_SIZE_serial 128
// uint8_t frameBuf_serial[MAX_FRAME_SIZE_serial];
// size_t  frameHead_serial     = 0;
// bool    frameReceived_serial = false;








//*********************** VLC FRAME ********************************************************

#define CMD_VLC_FRAME_DATA_REQ   "#VLD"// request data from node using optimised frame structure. 
//e.g.,  #VLD, Address_Destination (e.g 01,02,etc), CMD (e.g. 06,07,08)
// where CMD 06 means request sensor data only
// where CMD 07 means request location data only
// where CMD 08 means request both sensor & location data 
//*************************************************************************************************

// Function Declarations
void handleWakeup();
void handleSendVLCChar(String param);
void handleReceiveVLCChar(int duration);
void handleListenBLEAdvertise(int duration);


unsigned long advertisingStartTime;


int TxPower = 4;//TX=-20dBm, -16dBm, -12dBm, -8dBm, -4dBm, 0dBm, +3dBm and +4dBm.
int advertising_interval = int(40/0.625); //unit is 0.645fast:20-100ms, slow:100-2000ms. if set up 20ms, it is equal to 20/0.625
int Connection_Interval = int(45/1.25); // unit is 1.25ms, if set up 45ms, it is equal to 45/1.25

bool inFastMode = true;
int Advertising_fast_time = 30;

int phy_rate  = PHY_1M; //1Mbps, 2Mbps, 500 kbps/125 kbps
int mtu_value = 31; //23-247bytes

void setup() {
  collected_data.running_state = 0;
  // Initialize Serial Communication
  Serial.begin(BAUD_RATE);
  while (!Serial); // Wait for Serial to initialize
  Serial.println("MINILAMP 2025.10.30");
  Serial.println("-----------------------");

  // senhui initize the structure
    // collected_data.running_state             = 0;
    collected_data.wakeup_state           = 1;
    collected_data.communication_mode     = 0;
    collected_data.running_mode           = 0;

    collected_data.vlc_protocol_mode      = 2;
    collected_data.vlc_interval_seconds   = TIMER0_INTERVAL_MS / 10;
    collected_data.vlc_communication_volume = 0;
    collected_data.vlc_information_volume = 0;
    collected_data.vlc_per_unit_time      = 20;
    collected_data.pwm_percentage         = 90;

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



  bles_setup();
  nbvlc_setup();
  nbvlc_lamp_p((float)collected_data.pwm_percentage);  
  Serial.println("Module Ready"); // Indicate module is ready
  nbvlc_off();
}



//AA vlc-ble cmd data1 data2 data3 ... data n EE. E.g., AA 02 04  1e 11 21 EE


//////////////////////ble/////////////////
  // unsigned long vlc_sent = 0;
  static bool auto_send_vlc_flag;
  static uint8_t vlc_send_data[6];

void vlc_sendPeriodicFrames(uint8_t address, unsigned long interval_s, unsigned long count, uint8_t* data) {
  static unsigned long vlc_nextSendTime = 0; // only set once
  static unsigned long vlc_sent = 0;

  if (vlc_sent < count) {
    unsigned long now = millis();
    if (now >= vlc_nextSendTime) {
      // sendFrame(address);
      sendFrame(data, 0); // Send data
      vlc_sent++;
      vlc_nextSendTime = now + interval_s * 1000UL;  // schedule next send

      Serial.print("[Done] vlc_Sent ");
      Serial.print(vlc_sent);
      Serial.println(" frames");
    }
  } else {
        // CLear bit 1 
    collected_data.running_mode &= ~(1 << 1);
    send_periodic_frame_example();
    auto_send_vlc_flag = false;
    vlc_sent = 0;              // reset for next use
    vlc_nextSendTime = 0;      // reset timer
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
    vlc_sendPeriodicFrames(vlc_address, vlc_interval_s, vlc_duration_s, vlc_send_data);
  }

}

void handleVLC(uint8_t cmd, uint8_t* data1)
{
  if (cmd == CMD_RECEIVE)
  {
    // sendFrame(data1[0]); // Send data to destination node
    sendFrame(data1, 0); // Send data
    Serial.println("Send frame to node !");
    delay(100);
  }

  else if (cmd == CMD_Running_State)
  {
    collected_data.running_state = data1[0];
  }
  else if (cmd == CMD_Wakeup)
  {
    collected_data.wakeup_state = data1[0];
    if (collected_data.wakeup_state == 0)// go to sleep
    {
      // deep_sleep_process();
      ;
    }
  }
  else if (cmd == CMD_Communcation_mode)
  {
    collected_data.communication_mode = data1[0];
  }
  else if (cmd == CMD_VLC_Protocol)
  {
    collected_data.vlc_protocol_mode = data1[0]; 
    if ((collected_data.vlc_protocol_mode != 1)|(collected_data.vlc_protocol_mode != 2))
    {
      collected_data.vlc_protocol_mode == 1;
    }

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
    setInterval_time0(collected_data.vlc_interval_seconds);      
  }
  else if (cmd == CMD_PWM)
  {
    collected_data.pwm_percentage = data1[0]; 

      if(collected_data.pwm_percentage < 0) 
      { 
        collected_data.pwm_percentage == 0;
      }
      else if (collected_data.pwm_percentage > 101) 
      { 
        collected_data.pwm_percentage == 100;
      }

      if( (collected_data.pwm_percentage>=0) && (collected_data.pwm_percentage < 101) )
      {
        nbvlc_lamp_p((float)collected_data.pwm_percentage);   
      }
       
  }
  else if (cmd == CMD_VLC_Unit)
  {
    collected_data.vlc_per_unit_time = data1[0];
    Serial.print("vlc_per_unit_time:");
    Serial.println(collected_data.vlc_per_unit_time);  
  }
  else if (cmd == CMD_VLC_SEND_Auto)//const uint8_t CMD_VLC_SEND_Auto = 0x10; //SEND VLC AUTO FOR TEST //e.g., AA 01 10 05 20 01 EE
  {
    vlc_interval_s = data1[0];
    vlc_duration_s = data1[1];
    vlc_address   = data1[2];
    vlc_send_data[0] = vlc_address;//cmd
    vlc_send_data[1] = data1[3];//cmd
    auto_send_vlc_flag = true; 
  }
    else if (cmd == CMD_VLC_OFF_ON)//set vlc off 0 or on 1
  {
    if (data1[0] == 0)
    {
      handleVLCLampOFF(); 
      collected_data.pwm_percentage = 0; 
    // CLear bit 3 
      collected_data.running_mode &= ~(1 << 3);
      send_periodic_frame_example(); //send node status
      Serial.print("VLC OFF:");
    }
    else if(data1[0] == 1)
    {
      handleVLCLampON(93); 
      collected_data.pwm_percentage = 93; 
      collected_data.running_mode |= (1 << 3); 
      send_periodic_frame_example(); //send node status
      Serial.print("VLC ON:");
    }
  }

}


//////////////////////ble/////////////////
void ble_Scanner_connect()
{
    unsigned long startTime = millis(); // Record the start time
    BLE_command_received = true; // scan the command flag
    scan_type = SCAN_TYPE_WRITE; // for compatibility with scan_callback() and make it work and scan
    update_char = 1; // for compatibility with scan_callback() and make it work and scan

    if(!isConnected)//if connecting node, donot need to connect again.
    {
      Bluefruit.Scanner.start(0);
      Bluefruit.Advertising.start(0);
      Serial.println("Star to scan the ble node ... "); 
      while (millis() - startTime < BLE_WRITE_DURATION * 1000) {
        //delay(1000); // Short delay to avoid a tight loop
        if(isConnected==true)
        {
          Serial.println("Connect the ble node successfully! "); 
          break;
        }
      } 
    }
}

static bool auto_send_ble_flag;
static uint8_t ble_send_data[16];

void ble_sendPeriodicFrames(uint8_t* address, unsigned long interval_s, unsigned long count, uint8_t* data) {
  static unsigned long ble_nextSendTime = 0; // only set once
  static unsigned long ble_sent = 0;

  if (ble_sent < count) {
    unsigned long now = millis();
    if (now >= ble_nextSendTime) {

      sendBLEFrame(data,0);
      // send_respond_BLEFrame(address);
      ble_Scanner_connect();
      ble_sent++;
      ble_nextSendTime = now + interval_s * 1000UL;  // schedule next send

      Serial.print("[Done] ble_sent ");
      Serial.print(ble_sent);
      Serial.println(" frames");
    }
  } else {
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
    ble_sendPeriodicFrames(ble_address, ble_interval_s, ble_duration_s, ble_send_data);
  }

}

void handleBLE(uint8_t cmd, uint8_t* data1)
{

    if (cmd == CMD_RECEIVE)
    {
      sendBLEFrame(data1, 0);
      ble_Scanner_connect();
    }

    else if (cmd == CMD_Running_State)
    {
      collected_data.running_state = data1[0];
    }
    else if (cmd == CMD_Wakeup)
    {
      collected_data.wakeup_state = data1[0];
      Serial.print("BLE wakeup_state:");
      Serial.println(collected_data.wakeup_state);
      if (collected_data.wakeup_state == 0)// go to sleep
      {
        // deep_sleep_process();
        ;
      }
    }
    else if (cmd == CMD_Communcation_mode)
    {
      collected_data.communication_mode = data1[0];
    }
    else if (cmd == CMD_BLE_Protocol)
    {
      collected_data.ble_protocol_mode = data1[0];            
    }
    else if (cmd == CMD_BLE_Intervel)
    {
      collected_data.ble_interval_seconds = data1[0];
      Serial.print("ble_interval_seconds:");
      Serial.println(collected_data.ble_interval_seconds);
      CMD_BLE_Intervel_Process();

    }
    else if (cmd == CMD_PHY_Rate)
    {
      collected_data.phy_rate_percentage = data1[0];
      Serial.print("phy_rate_percentage:");
      Serial.println(collected_data.phy_rate_percentage);  
      CMD_PHY_Rate_Process(); 
    }
    else if (cmd == CMD_MTU)
    {
        collected_data.mtu_value = data1[0];
        Serial.print("mtu_value:");
        Serial.println(collected_data.mtu_value); 
        CMD_MTU_Process();
    }
    else if (cmd == CMD_BLE_Unit)
    {
      collected_data.ble_per_unit_time = data1[0];
      Serial.print("ble_per_unit_time:");
      Serial.println(collected_data.ble_per_unit_time);  
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

      ble_send_data[0]   = data1[2];//add0
      ble_send_data[1]   = data1[3];
      ble_send_data[2]   = data1[4];
      ble_send_data[3]   = data1[5];
      ble_send_data[4]   = data1[6];
      ble_send_data[5]   = data1[7];//add5
      ble_send_data[6]   = data1[8];//cmd/data0
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
        send_periodic_frame_example(); //send node status
      }
      else if(data1[0] == 1)
      {
        // Mode 1: BLE Idle (enabled but not advertising)
        Bluefruit.begin();  // Re-enable BLE if it was off
        Bluefruit.Advertising.stop();
        Serial.println("BLE enabled, not advertising");
        // Set bit 2 of running_mode
        collected_data.running_mode |= (1 << 2); 
        send_periodic_frame_example(); //send node status
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


unsigned long  BLE_start_connection_time=0;//avoid connecting teh node for long time. e.g., received invalid frame or no response. which cannot make BLE_command_received = false;  // disconnect the node 
bool record_time_flag = false;
int max_connection_time = 20000;

void ble_auto_disconnect()
{
  // Once the gateway has finished processing the frame, ensure it is disconnected
  if (!BLE_command_received && isConnected) {
    Bluefruit.disconnect(current_conn_handle); // Disconnect the client
    bles_stop_scan();
    current_conn_handle = BLE_CONN_HANDLE_INVALID; // Reset connection handle
  }

  if(isConnected && !record_time_flag)
  {
    BLE_start_connection_time = millis(); // Record the start time
    record_time_flag = true;
    // Set bit 5 of running_mode
    collected_data.running_mode |= (1 << 5); 
    send_periodic_frame_example(); //send node status
  }
  if (((millis() - BLE_start_connection_time) > max_connection_time)&&(isConnected))
  {
    BLE_command_received = false; // force stop connection
    BLE_start_connection_time = millis(); // Record the start time again to avoid running this if.
    // CLear bit 5 
    collected_data.running_mode &= ~(1 << 5);
    send_periodic_frame_example(); //send node status
    Serial.println("Force stop connection due to connect the node for long time!"); // Print a newline after the loop
  }
}


void check_ble_inactivity_disconnect() 
{
  if (isConnected && ((millis() - last_ble_activity_time) > BLE_IDLE_TIMEOUT)) 
  {
    Serial.println("Disconnecting due to BLE inactivity.");
    scan_type = SCAN_TYPE_NULL;
    Bluefruit.disconnect(current_conn_handle); 
    Bluefruit.Scanner.stop();
    Bluefruit.Advertising.stop();
    current_conn_handle = BLE_CONN_HANDLE_INVALID;
    isConnected = false;

  }
}



void loop() {
  collected_data.running_state = 2;
  serial_processing();// serial command processing
  auto_send_ble();
  auto_send_vlc();
  ble_auto_disconnect();
  // check_ble_inactivity_disconnect();

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


// void loop() {

// //************************* ADD LISTENING VLC FRAMES *****************************
// processReceivedData();
// //***************************************************************************************


//   /*****************VLC Senhui Add**********************/
//   processFrame();
//     if (frameFlag) {
//         Serial.println("Valid frame received:");
//         for (size_t i = 0; i < frameIndex_clone; i++) {
//             if (frameBuffer[i] < 0x10) {
//                 Serial.print("0"); // Add leading zero for values less than 0x10
//             }
//             Serial.print(frameBuffer[i], HEX);
//             Serial.print(" "); // Add a space between values
//         }
//         Serial.println(); // Print a newline after the loop
//         frameIndex_clone = 0;  // Reset frameIndex_clone
//         frameFlag = false;  // Reset the flag
//     }
//   /*****************BLE Senhui Add**********************/
//   // Once the gateway has finished processing the frame, ensure it is disconnected
//   if (!BLE_command_received && isConnected) {
//     Bluefruit.disconnect(current_conn_handle); // Disconnect the client
//     bles_stop_scan();
//     current_conn_handle = BLE_CONN_HANDLE_INVALID; // Reset connection handle
//   }

//   if(isConnected && !record_time_flag)
//   {
//     BLE_start_connection_time = millis(); // Record the start time
//     record_time_flag = true;
//   }
//   if (((millis() - BLE_start_connection_time) > max_connection_time)&&(isConnected))
//   {
//     BLE_command_received = false; // force stop connection
//     BLE_start_connection_time = millis(); // Record the start time again to avoid running this if.
//     Serial.println("Force stop connection due to connect the node for long time!"); // Print a newline after the loop
//   }
//   /*****************END Senhui Add**********************/






//   // Check if there's incoming data on Serial
//   if (Serial.available()) {
//     String command = Serial.readStringUntil('\n'); // Read the command from Serial
//     command.trim(); // Remove whitespace characters

//     // Parse and handle commands
//     if (command == CMD_WAKEUP) {
//       handleWakeup();
//     } 
//     else if (command.startsWith(CMD_SEND_VLC_CHAR)) {
//       String param = command.substring(strlen(CMD_SEND_VLC_CHAR) + 1); // Extract parameter after "#VLS,"
//       handleSendVLCChar(param);
//     } 
//     else if (command.startsWith(CMD_RECEIVE_VLC_CHAR)) {
//       String param = command.substring(strlen(CMD_RECEIVE_VLC_CHAR) + 1); // Extract duration after "#RVC,"
//       int duration = param.toInt();
//       handleReceiveVLCChar(duration);
//     } 
//     else if (command.startsWith(CMD_LISTEN_BLE_ADVERTISE)) {
//       String param = command.substring(strlen(CMD_LISTEN_BLE_ADVERTISE) + 1); // Extract duration after "#LBA,"
//       int duration = param.toInt();
//       handleListenBLEAdvertise(duration);
//     }
//     else if (command.startsWith(CMD_WRITE_BLE_CHAR)) {
//       // Find the comma separator
//       String mac_str;
//       uint8_t hex_value;
//       String param = command.substring(strlen(CMD_LISTEN_BLE_ADVERTISE) + 1); // Extract duration after "#WBL,"
//       int commaIndex = param.indexOf(',');

//       // Extract the UUID substring
//       if (commaIndex != -1) {
//         mac_str = param.substring(0, commaIndex);

//         // Extract and convert the 2-digit hex value
//         String hex_str = param.substring(commaIndex + 1);
//         hex_value = (uint8_t) strtol(hex_str.c_str(), nullptr, 16);
//       }
//       handleWriteBLECharacteristic(mac_str, hex_value);
//     }
//     else if (command.startsWith(CMD_VLC_LAMP_OFF)) {
//       handleVLCLampOFF();
//     }    
//     else if (command.startsWith(CMD_VLC_LAMP_ON)) {
//       String param = command.substring(strlen(CMD_VLC_LAMP_ON) + 1); // Extract duration after "#LBA,"
//       int power = param.toInt();
//       handleVLCLampON(power);
//     }   
//     else if (command.startsWith(CMD_VLC_LAMP_P)) {
//       String param = command.substring(strlen(CMD_VLC_LAMP_P) + 1); // Extract duration after "#LBA,"
//       int power = param.toInt();
//       handleVLCLampP(power);
//     } 
//     else if (command.startsWith(CMD_HELP)) {
//       printCommandTable();
//     } 
//     /***************** VLC Frame **********************/
//     else if (command.startsWith(CMD_VLC_FRAME_DATA_REQ)) 
//     {
//     if (!processSerialCommand(command)) 
//     {
//     }
//     }

//     /*****************VLC Senhui Add**********************/
//     else if (command.startsWith(CMD_SEND_VLC_FRAME)) {
//         // Extract the parameters after "#VLF,"
//         String param = command.substring(0); // Skip "#VLF,"
//         uint8_t* send_data = nullptr;
//         int count = 0;

//         // Call the function to extract data
//         extractDataFromCmd(param, &send_data, count);
//         //Debug for printing extract data
//         for (int i = 0; i < count; i++) {
//           Serial.print("0x");
//           Serial.print(send_data[i], HEX);
//           Serial.print(" ");
//         }
//         Serial.println();

//         // Call the handling function
//         handleSendVLCFrame(send_data, count);
//         free(send_data);
//     }
//     /*****************BLE Senhui Add**********************/
//     else if (command.startsWith(CMD_SEND_BLE_FRAME)) {
//         // Extract the parameters 
//         String param = command.substring(0); 
//         // Pointer for the extracted data
//         uint8_t* send_data = nullptr;
//         int data_len = 0;

//         // Call the function to extract data
//         extractDataFromCmd(param, &send_data, data_len);
//         //Debug for printing extract data
//         for (int i = 0; i < data_len; i++) {
//           Serial.print("0x");
//           Serial.print(send_data[i], HEX);
//           Serial.print(" ");
//         }
//         Serial.println();

//         //replace ble mac with extract mac
//         for (uint8_t i=0; i<6; i++)
//         {
//           target_mac[i] = send_data[i];
//           // Serial.print("0x");
//           // Serial.print(target_mac[i], HEX);
//           // Serial.print(" ");
//         }
//         unsigned long startTime = millis(); // Record the start time

//         scan_type = SCAN_TYPE_WRITE; // for compatibility with scan_callback() and make it work and scan
//         update_char = 1; // for compatibility with scan_callback() and make it work and scan

//        int length_data = data_len-7;//  6 mac, 1 cmd
//       if (length_data<0) length_data = 0;

//         Serial.print("length_data: ");
//         Serial.println(length_data);
//         for (uint8_t i=0; i<(6+1+length_data); i++)
//         {
//           Serial.print("0x");
//           Serial.print(send_data[i], HEX);
//           Serial.print(" ");
//         }
//         Serial.println();
 
//         handleSendBLEFrame(send_data, length_data);
//         BLE_command_received = true; // scan the command flag
//         free(send_data);

//         if(!isConnected)//if connecting node, donot need to connect again.
//         {
//           Bluefruit.Scanner.start(0);
//           while (millis() - startTime < BLE_WRITE_DURATION * 1000) {
//             //delay(1000); // Short delay to avoid a tight loop
//             if(isConnected==true)
//             break;
//           } 
//         }
//     }


//     else {
//       Serial.println("Error 1: Invalid Command");
//     }
//   }
// }

// Command Handlers
void handleWakeup() {
  Serial.println("Ok");
}

void handleSendVLCChar(String param) {
  if (param.length() == 2) { // Check if the parameter is a 2-digit hex
    // Convert the hex string to an 8-bit unsigned integer
    uint8_t value = (uint8_t) strtoul(param.c_str(), NULL, 16);

    if( nbvlc_send(value) ) Serial.println("Ok");
  } else {
    Serial.println("Error: Invalid Character Format");
  }
}


void handleSendVLCFrame(uint8_t* send_data, int count) {
  if (send_data != nullptr && count > 0) { 
    sendFrame(send_data, count); // Senhui add;
    Serial.println("OK. VLC is going to send frame!");
  } else {
    Serial.println("Error: Invalid Frame Format");
  }
}


void handleSendBLEFrame(uint8_t* send_data, int count) {
  if (send_data != nullptr && count >= 0) { 
    sendBLEFrame(send_data, count); // Senhui add;
    ble_Scanner_connect(); //senhui add
    Serial.println("OK. BLE is going to send frame!");
  } else {
    Serial.println("Error: Invalid BLE Frame Format");
  }
}

void handleReceiveVLCChar(int duration) {
  unsigned long startTime = millis(); // Record the start time
  bool received = false;
  uint8_t receivedChar = 0x9A; // Simulated received character in hex format

  // Loop for the specified duration or until nbvlc_rx_run() returns true
  while (millis() - startTime < duration * 1000) {
    if (nbvlc_rx_run()) { // Check if VLC data has been received
      received = true;
      break;
    }
    //delay(1); // Short delay to avoid a tight loop
  }

  // Respond with either the received character or NULL if timeout
  Serial.print("Ok,");
  if (received) {
    receivedChar = nbvlc_get_data();
    if (receivedChar < 0x10) {
      Serial.print("0"); // Ensure 2-digit format by adding leading zero if necessary
    }
    Serial.println(receivedChar, HEX); // Print in hex format
  } else {
    Serial.println("NULL");
  }
}

void handleListenBLEAdvertise(int duration) {
  // Simulating a delay for listening
  /*
  delay(duration * 1000); // Simulated listening duration
  
  // Simulate UUIDs and Manufacturer Data
  String UUIDs = "UUID1,UUID2,UUID3,UUID4";
  String manufData = "MFD1,MFD2,MFD3,MFD4";

  Serial.print("Ok,");
  Serial.print(UUIDs);
  Serial.print(",");
  Serial.println(manufData);
  */
  unsigned long startTime = millis(); // Record the start time
  bles_start_scan(0);
  // Loop for the specified duration or until nbvlc_rx_run() returns true
  while (millis() - startTime < duration * 1000) {
    //delay(1000); // Short delay to avoid a tight loop
  }
  bles_stop_scan();
  Serial.println("Ok");
}

void handleWriteBLECharacteristic(String mac, uint8_t data) {
 
  unsigned long startTime = millis(); // Record the start time
  bles_start_scan(0, SCAN_TYPE_WRITE, &mac, data);  // Loop for the specified duration or until nbvlc_rx_run() returns true
  while (millis() - startTime < BLE_WRITE_DURATION * 1000) {
    //delay(1000); // Short delay to avoid a tight loop
  }
  bles_stop_scan();
  if( bles_written() ) Serial.println("Ok"); else Serial.println("Error: data not written.");
}

void handleVLCLampOFF() {
  nbvlc_lamp_off();
  Serial.println("Ok");
}

void handleVLCLampON(int power) {
  if( (power>=0) && (power < 101) )
  {
    nbvlc_lamp_on((float)power);
    Serial.println("Ok");
  } else
    Serial.println("Error: Power out of range 0 - 100");

}

void handleVLCLampP(int power) {
  if( (power>=0) && (power < 101) )
  {
    nbvlc_lamp_p((float)power);
    Serial.println("Ok");
  } else
    Serial.println("Error: Power out of range 0 - 100");
}

void printCommandTable() {
  Serial.println("Communication Specification");
  Serial.println("Hybrid VLC and BLE Communication Module");
  Serial.println("a.k.a the Mini Lamp v. 1.0");
  Serial.println("");
  Serial.println("Command\t\t\t\tShort Form Syntax\t\t\tExpected Response");
  
  Serial.print("Global Wakeup\t\t");
  Serial.print("#WUP\t\t\t\t");
  Serial.println("Ok");

  Serial.print("Send VLC Character\t");
  Serial.print("#VLS,<character>\t\t");
  Serial.println("Ok or Error: Invalid Character Format");

  Serial.print("Send VLC Frame\t\t");
  Serial.print("#VLF,<addr_destination,cmd, data0,...>\t");
  Serial.println("Ok or Error: Invalid Frame Format");

  Serial.print("Send optimised VLC Frame\t\t");
  Serial.print("#VLD,<addr_destination (e.g. 01,02,etc),cmd (e.g. 06,07,08)>\t");
  Serial.println("Ok or Error: Invalid Frame Format");


  Serial.print("Send BLE Frame\t\t");
  Serial.print("#BLF,<mac[6],cmd,data0,...>\t");
  Serial.println("Ok or Error: Invalid Frame Format");


  Serial.print("Receive VLC Character\t");
  Serial.print("#RVC,<duration>\t\t\t");
  Serial.println("Ok,<character> or Ok,NULL");

  Serial.print("Receive VLC Frame\t");
  Serial.print("#RVF,<duration>\t\t\t");
  Serial.println("Ok,<Frame> or Ok,NULL");

  Serial.print("Listen to BLE Adv\t");
  Serial.print("#LBA,<duration>\t\t");
  Serial.println("\t<UUID>,<RSSI>,<Mfr Data>");
  Serial.println("\t\t\t\t\t\t\t<UUID>,<RSSI>,<Mfr Data>");
  Serial.println("\t\t\t\t\t\t\t<UUID>,<RSSI>,<Mfr Data>");
  Serial.println("\t\t\t\t\t\t\t<UUID>,<RSSI>,<Mfr Data>");
  Serial.println("\t\t\t\t\t\t\t<UUID>,<RSSI>,<Mfr Data>");
  Serial.println("\t\t\t\t\t\t\t      ...");
  Serial.println("\t\t\t\t\t\t\tOk");

  Serial.print("Turn VLC Lamp Off\t");
  Serial.print("#LOF\t\t\t\t");
  Serial.println("Ok");

  Serial.print("Turn VLC Lamp On\t");
  Serial.print("#LON,<level>\t\t\t");
  Serial.println("Ok or Error: Power out of range 1 - 90");

  Serial.print("Set VLC Lamp Power\t");
  Serial.print("#LAP,<level>\t\t\t");
  Serial.println("Ok or Error: Power out of range 1 - 90");

  Serial.print("Write to BLE Char\t");
  Serial.print("#WBL, <MAC>,<value>\t\t");
  Serial.println("Ok or Error: data not written");
}

// // Create and send frame
// Frame frame;

bool processSerialCommand(uint8_t cmd, uint8_t* data1) {
      // Create and send frame
      Frame frame;
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

            uint8_t destAddr = 0x01; 
            frame.startMarker = START_MARKER;
            frame.sourceAddr  = MY_ADDRESS;  // Your device address
            frame.destAddr    = destAddr;
            frame.command     = CMD_REQ_BOTH;
            
            // Set payload size based on command type
            switch(frame.command) {
                case CMD_REQ_SENSOR:
                    frame.payloadSize = 0;
                    Serial.println("Requesting sensor data...");
                    break;
                case CMD_REQ_LOC:
                    frame.payloadSize = 0;
                    Serial.println("Requesting location data...");
                    break;
                case CMD_REQ_BOTH:
                    frame.payloadSize = 0;
                    Serial.println("Requesting sensor and location data...");
                    break;
                default:
                    frame.payloadSize = 0;
                    break;
            }
            
            frame.checksum = calculateChecksumFrame(&frame);
            frame.endMarker = END_MARKER;
            
            // Print debug info
            Serial.print("Sending frame to address 0x");
            Serial.print(destAddr, HEX);
            Serial.print(" with command 0x");
            Serial.println(cmd, HEX);
            
            // Transmit the frame
            transmitFrame(frame);
            return true;
}



// ********************************************************************************************
