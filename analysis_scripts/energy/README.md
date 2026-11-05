<!--
Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
SPDX-License-Identifier: BSD-3-Clause
-->

# Energy Estimator

Turns measured latency into energy per inference and picks the best
(model, board, threads) pair for a sensing spec.

**Hardware requirement: none** (reads existing summary CSVs).

- [`board_power.yaml`](board_power.yaml): `P_active`, `P_listen`, `P_sleep`
  per board as `[low, high]` watt ranges, each with sources and a confidence
  note.
  - Pi 4: `P_active` is given separately for 1 and 4 threads, and
    `P_sleep = P_listen` because the board has no whole-board sleep.
  - Nano RP2040 Connect: a board-level estimate built from component
    datasheets, to be replaced by a USB power measurement.
  - XIAO ESP32-S3 Sense: a board-level estimate from the ESP32-S3 datasheet
    (240 MHz, radios off) and Seeed's spec table, to be replaced by a USB
    power measurement.
- [`estimate_energy.py`](estimate_energy.py): reads `rp2040_summary.csv` and
  `rpi4_summary.csv` and applies

  ```
  S <= L: E = P_active * t + P_listen * (S - t)
  S >  L: E = P_active * t + P_listen * L + P_sleep * (S - L - t)
  ```

  A pair counts only if its input window equals `L`. It is infeasible if:
  - its benchmark status is not `ok`;
  - `t > S`;
  - for `S > L`, `t > S - L`. This is a **conservative assumption**: compute
    does not overlap listening, so each period is listen `L`, compute `t`,
    then sleep. Revisit it if a pair ever fails only this rule;
  - `t > R` when a maximum response latency `R` is given (`--max-response`).
    `R` is separate from the stride. Response latency is currently the
    `invoke()` time only; on-device log-mel preprocessing will add to it.
  It writes `results/energy_estimates.csv` (every pair, low/high energy and
  average power) and `results/energy_selection.md` (per spec: all pairs, then
  winner and runner-up for each minimum **validation-fold** accuracy, low vs.
  high power), starting with a compact winners table over all specs and
  response limits.

- [`preprocessing.yaml`](preprocessing.yaml): measured log-mel times per
  (board, threads): one frame (= one 10 ms hop) and whole 1 s / 2 s windows.
  Unmeasured entries are marked `pending` and excluded, never treated as zero.
- `--design` chooses how preprocessing is accounted:
  - `inference_only`: `t` = inference only (the earlier results);
  - `naive`: the whole window's log-mel runs after the window ends,
    `t = window time + inference`;
  - `streaming`: frames are computed as audio arrives, so
    `t = one frame + inference`, and listening power becomes
    `P_listen + duty * (P_active - P_listen)` with `duty = frame / hop`.

  With several designs, the report also lists every cell whose winner
  differs from `inference_only`.

```bash
python analysis_scripts/energy/estimate_energy.py                                   # default specs
python analysis_scripts/energy/estimate_energy.py --spec 1,1 2,4 --min-acc 0.67     # custom
python analysis_scripts/energy/estimate_energy.py --max-response none 0.2 0.5 1.0  # response limits
python analysis_scripts/energy/estimate_energy.py --design inference_only streaming naive \
  --max-response none 0.2 0.5 1.0                                                   # with preprocessing
```

## Power measurement (INA228 on VBUS, read by the Pi 4)

This replaces the datasheet ranges with board-level measurements. Nothing
here needs sudo except enabling I2C on the Pi.

### Wiring

```
laptop USB-C ── laptop-side breakout ── VBUS ─► INA228 VIN+ ─[15 mΩ]─ VIN- ─► VBUS ── Adafruit 5978 ── USB-C→micro-USB ── board
                                        D+ / D- / GND / CC ─────────────── straight through ──────────────
INA228 VCC/SDA/SCL/GND ── Pi 4 header pin 1 (3.3 V) / 3 (SDA) / 5 (SCL) / 6 (GND); Pi GND common with breakout GND
```

- Only VBUS goes through the shunt. D+, D−, GND and the CC line(s) pass
  straight through, so the board still enumerates, uploads and prints serial.
  Without CC (or a 5.1 kΩ pull-down from CC to GND on the board side) the
  laptop won't enable VBUS; USB-C to micro-USB adapters normally include it.
- The INA228's VBUS sense pin must see the load side (VIN−). Check the
  breakout's VBUS jumper.
- Keep the D+/D− jumpers short and together (both boards are USB full speed).

### Enable I2C on the Pi and find the INA228 (run these yourself)

```bash
ssh rpi4@<pi-host>
sudo raspi-config nonint do_i2c 0       # or: sudo raspi-config → Interface Options → I2C → Yes
ls /dev/i2c-1                           # should exist (reboot if it does not)
groups | grep -w i2c                    # user must be in the i2c group
sudo apt install i2c-tools              # only if i2cdetect is missing
i2cdetect -y 1                          # expect "40" (INA228 default; A0/A1 straps give 0x40-0x4F)
```

The reader checks the chip IDs (manufacturer `0x5449`, device `0x228`)
before measuring and refuses anything else. It needs only Python 3, with no
packages.

### Firmware (host)

`prepare_power_firmware.py` builds nine variants of the latency sketch with
the SC_5 1s model and `CREST_POWER_STATE`:

| variant | state (`CREST_POWER_STATE`) |
|---|---|
| `rp2040_inference`, `rp2040_inference_nina_reset` | 1: continuous `Invoke()`; NINA free / held in reset |
| `rp2040_listen`, `rp2040_listen_nina_reset` | 2: **listening = P_listen**: MP34DT06J PDM mic capturing 16 kHz mono (PDM library, PIO/DMA), CPU otherwise idle, no inference |
| `rp2040_idle_mic_off`, `rp2040_idle_mic_off_nina_reset` | 3: CPU idle (`delay()`), mic off |
| `esp32s3_inference`, `esp32s3_listen`, `esp32s3_idle_mic_off` | the same three states at 240 MHz, Wi-Fi/BT off; the listening state uses the Sense board's PDM mic via `ESP_I2S` PDM RX (GPIO42 clock, GPIO41 data, hardware PDM->PCM) |

The microphone's own cost is `listen` minus `idle_mic_off`. In the listening
state the END line also reports `mic_samples` (about 16,000 per second
expected), `mic_rate_hz` and `mic_peak`, so you can confirm the mic ran.

After a 10 s settle, every variant repeats 60 s holds forever. Each hold
starts with one LED blink and a `POWER_STATE_BEGIN ...` serial line, and ends
with a `POWER_STATE_END ... invokes=N mean_invoke_us=...` line and two
blinks. The LED is off during the hold.

```bash
python analysis_scripts/energy/prepare_power_firmware.py                              # compile all
python analysis_scripts/energy/prepare_power_firmware.py --upload rp2040_inference_nina_reset --port /dev/ttyACM0
```

### Procedure (per variant)

1. **Offset check (once per session):** with nothing on the board side, run
   one 10 s reader window. `amps_mean` should be within a few current LSBs
   (5.2 µA each) of zero; record the value as a `--note` if it is not.
2. Flash the variant through the inline setup. Optionally watch the serial
   port to see `POWER_STATE_BEGIN` (e.g. `python -m serial.tools.miniterm /dev/ttyACM0 115200`).
3. **Wait** 10 s (firmware settle) plus at least 30 s for temperature, i.e.
   start the reader with `--wait-s 30` right after the first `POWER_STATE_BEGIN`.
4. Measure **3 windows of 60 s** (`--windows 3 --seconds 60`):
   ```bash
   scp analysis_scripts/energy/ina228_reader.py rpi4@<pi-host>:crest_rpi4/
   ssh rpi4@<pi-host> 'python3 crest_rpi4/ina228_reader.py --windows 3 --seconds 60 --wait-s 30 --label rp2040_inference_nina_reset' | tee rp2040_inference_nina_reset.txt
   ```
5. **Record** per window: bus voltage, mean current (with std, min, max),
   mean power, sample count and rate, die temperature, shunt value and ADC
   range (all printed by the reader). Also record the meter details: the
   INA228 breakout model, shunt (15 mΩ unless the breakout differs), and the
   resolution, 5.21 µA = about 26 µW at 5 V per LSB (`power_resolution_w`).
   Note the ambient temperature and which laptop port was used.
6. Store them:
   ```bash
   python analysis_scripts/energy/record_power_measurement.py --board rp2040 --state inference_nina_reset \
       --from-file rp2040_inference_nina_reset.txt --note "Adafruit INA228, 15 mOhm, laptop USB-C port 1, 23 C"
   ```
   States: rp2040 `inference_nina_free`, `listen_nina_free`,
   `idle_mic_off_nina_free`, `inference_nina_reset`, `listen_nina_reset`,
   `idle_mic_off_nina_reset`; esp32s3 `inference`, `listen`, `idle_mic_off`
   (the selection uses `inference` as P_active and `listen` as P_listen;
   `idle_mic_off` is for comparison); rpi4 `idle`, `active_1t`, `active_4t` (optional: the Pi can be
   powered through the same INA228 while reading it).
7. Rerun the selection from the measured values (datasheet ranges stay in
   `board_power.yaml` for comparison):
   ```bash
   python analysis_scripts/energy/estimate_energy.py --power-source measured \
     --design inference_only streaming naive --max-response none 0.2 0.5 1.0
   ```
   A measured value's range is [min − resolution, max + resolution] of its
   window means. Values without samples, and P_sleep (no sleep firmware
   yet), fall back to the datasheet; the report lists the source of every
   value. `nina_for_selection` in `board_power_measured.yaml` picks which
   RP2040 NINA state the selection uses.
