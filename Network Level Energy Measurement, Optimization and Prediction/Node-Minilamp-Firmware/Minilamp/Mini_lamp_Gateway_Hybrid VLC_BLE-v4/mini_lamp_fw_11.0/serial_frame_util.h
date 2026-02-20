#ifndef SERIAL_FRAME_UTIL_H
#define SERIAL_FRAME_UTIL_H

#include <Arduino.h> // Include Arduino.h for byte and size_t types


typedef struct CollectedData {
  byte running_state;
  byte wakeup_state;
  byte communication_mode;
  byte running_mode;

  byte vlc_protocol_mode;
  byte vlc_interval_seconds;
  byte vlc_communication_volume;
  byte vlc_information_volume;
  byte vlc_per_unit_time;
  byte pwm_percentage;

  byte ble_protocol_mode;
  byte ble_interval_seconds;
  byte ble_communication_volume;
  byte ble_information_volume;
  byte ble_per_unit_time;
  byte phy_rate_percentage;
  byte mtu_value;
  byte fast_time;

  byte spare_2;
  byte spare_3;
  byte spare_4;
} CollectedData_t; // Optional: Define a type alias for convenience


// Function prototype for creating a serial frame (scalable version)
byte* create_serial_frame(byte* data_bytes, size_t data_length);

// Function prototype for sending periodic frame example (NEW)
void send_periodic_frame_example(bool send_flag = false);
extern CollectedData collected_data;

#endif // SERIAL_FRAME_UTIL_H