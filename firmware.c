#include "xparameters.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xuartlite_l.h"
#define LED_GPIO_BASE 0x40010000U  // axi_gpio mapped at this base (write lower 4 bits)
#define HAS_XGPIO 0  // driver not generated; use raw MMIO

#define SYS_CTRL_BASE   0x00010000U
#define REG_CTRL        0x0
#define REG_STATUS      0x4
#define REG_STREAM_LEN  0x8
#define REG_FLUSH_LEN   0xC

#define BRAM_A_BASE     0x40000000U
#define BRAM_B_BASE     0xC0000000U
#define BRAM_C_BASE     0xC2000000U
#define UART_BASE       XPAR_AXI_UARTLITE_0_BASEADDR

#define OPC_PING     0x00
#define OPC_SET_LEN  0x01
#define OPC_LOAD_A   0x10
#define OPC_LOAD_B   0x11
#define OPC_RUN      0x20
#define OPC_READ_C   0x21
#define OPC_VERSION  0x30
#define STATUS_OK    0x00
#define STATUS_BADLEN 0xEE
#define STATUS_BADOP  0xEF

#define MAX_TILE_COUNT 32

static inline void write_reg(u32 off, u32 val) { Xil_Out32(SYS_CTRL_BASE + off, val); }
static inline u32  read_reg(u32 off)          { return Xil_In32(SYS_CTRL_BASE + off); }
#if HAS_XGPIO
#define LED_DEVICE_ID XPAR_AXI_GPIO_0_DEVICE_ID
#define LED_CHANNEL   1
static XGpio led_gpio;
#endif

static void init_leds(void) {
#if HAS_XGPIO
    XGpio_Initialize(&led_gpio, LED_DEVICE_ID);
    XGpio_SetDataDirection(&led_gpio, LED_CHANNEL, 0x0);
#else
    (void)0;
#endif
}

static void set_leds(u32 pattern) {
#if HAS_XGPIO
    XGpio_DiscreteWrite(&led_gpio, LED_CHANNEL, pattern & 0xF);
#else
    Xil_Out32(LED_GPIO_BASE, pattern & 0xF);
#endif
}

static void uart_send_byte(u8 b) { XUartLite_SendByte(UART_BASE, b); }
static u8 uart_recv_byte(void) { return XUartLite_RecvByte(UART_BASE); }

static void uart_send_buf(const u8 *buf, unsigned len) {
    for (unsigned i = 0; i < len; i++) uart_send_byte(buf[i]);
}

static int uart_recv_buf(u8 *buf, unsigned len) {
    for (unsigned i = 0; i < len; i++) buf[i] = uart_recv_byte();
    return 0;
}

static void put_u32le(u32 v, u8 *out) {
    out[0] = (u8)(v & 0xFF);
    out[1] = (u8)((v >> 8) & 0xFF);
    out[2] = (u8)((v >> 16) & 0xFF);
    out[3] = (u8)((v >> 24) & 0xFF);
}

static u32 get_u32le(const u8 *in) {
    return ((u32)in[0]) |
           ((u32)in[1] << 8) |
           ((u32)in[2] << 16) |
           ((u32)in[3] << 24);
}

// Pack a 4x4 matrix (row-major) into 8 time-step words for the systolic array.
// Each word packs {a4,a3,a2,a1} as bytes; lanes are offset by their row index.
static void pack_a_words(const u32 *mat, u32 *a_words) {
    for (int t = 0; t < 8; t++) {
        u32 w = 0;
        for (int lane = 0; lane < 4; lane++) {
            int col = t - lane;
            u8 v = (col >= 0 && col < 4) ? (u8)(mat[lane * 4 + col] & 0xFF) : 0;
            w |= ((u32)v) << (8 * lane);
        }
        a_words[t] = w;
    }
}

// Pack a 4x4 matrix (row-major) into 8 time-step words for the systolic array.
// Each word packs {b4,b3,b2,b1} as bytes; lanes are offset by their column index.
static void pack_b_words(const u32 *mat, u32 *b_words) {
    for (int t = 0; t < 8; t++) {
        u32 w = 0;
        for (int lane = 0; lane < 4; lane++) {
            int row = t - lane;
            u8 v = (row >= 0 && row < 4) ? (u8)(mat[row * 4 + lane] & 0xFF) : 0;
            w |= ((u32)v) << (8 * lane);
        }
        b_words[t] = w;
    }
}

// Simple software reference model of the 4x4 systolic array.
// Feeds the same stream/padding as hardware and returns 16 results in row-major order.
static void systolic_ref(const u32 *a_words, const u32 *b_words,
                         int stream_len, int flush_len, s32 *c_out) {
    s32 c[4][4] = {0};
    s8 a_out[4][4] = {{0}};
    s8 b_out[4][4] = {{0}};
    for (int phase = 0; phase < 1 + stream_len + flush_len; phase++) {
        s8 ext_a[4];
        s8 ext_b[4];
        int clear = (phase == 0);
        if (phase == 0) {
            for (int i = 0; i < 4; i++) { ext_a[i] = 0; ext_b[i] = 0; }
        } else if (phase >= 1 && phase <= stream_len) {
            int idx = phase - 1;
            for (int r = 0; r < 4; r++) ext_a[r] = (s8)((a_words[idx] >> (8*r)) & 0xFF);
            for (int ccol = 0; ccol < 4; ccol++) ext_b[ccol] = (s8)((b_words[idx] >> (8*ccol)) & 0xFF);
        } else {
            for (int i = 0; i < 4; i++) { ext_a[i] = 0; ext_b[i] = 0; }
        }

        s8 next_a_out[4][4];
        s8 next_b_out[4][4];
        for (int r = 0; r < 4; r++) {
            for (int ccol = 0; ccol < 4; ccol++) {
                s8 a_in = (ccol == 0) ? ext_a[r] : a_out[r][ccol-1];
                s8 b_in = (r == 0) ? ext_b[ccol] : b_out[r-1][ccol];
                if (clear) c[r][ccol] = 0;
                c[r][ccol] += (s32)a_in * (s32)b_in;
                next_a_out[r][ccol] = a_in;
                next_b_out[r][ccol] = b_in;
            }
        }
        for (int r = 0; r < 4; r++) {
            for (int ccol = 0; ccol < 4; ccol++) {
                a_out[r][ccol] = next_a_out[r][ccol];
                b_out[r][ccol] = next_b_out[r][ccol];
            }
        }
    }

    for (int r = 0; r < 4; r++)
        for (int ccol = 0; ccol < 4; ccol++)
            c_out[r*4 + ccol] = c[r][ccol];
}

int main(void) {
    xil_printf("Systolic array RPC firmware (UART)\r\n");
    init_leds();

    u32 mat_a[16] = {0};
    u32 mat_b[16] = {0};
    u16 stream_len = 8;
    u16 flush_len  = 10;
    u16 tile_count = 1;
    int tile_count_locked = 0; // locked when caller provides explicit tile_count

    while (1) {
        u8 hdr[2];
        uart_recv_buf(hdr, 2);
        u8 op = hdr[0];
        u8 len = hdr[1];

        switch (op) {
        case OPC_PING:
            if (len != 0) { uart_send_byte(STATUS_BADLEN); break; }
            uart_send_byte(STATUS_OK);
            break;
        case OPC_SET_LEN:
            if (len != 4 && len != 6) { uart_send_byte(STATUS_BADLEN); break; }
            {
                u8 buf[6];
                uart_recv_buf(buf, len);
                stream_len = (u16)get_u32le(buf);      // lower 16 bits
                flush_len  = (u16)(get_u32le(buf) >> 16);
                if (len == 6) {
                    tile_count = ((u16)buf[4]) | ((u16)buf[5] << 8);
                    if (tile_count == 0 || tile_count > MAX_TILE_COUNT) { uart_send_byte(STATUS_BADLEN); break; }
                    tile_count_locked = 1;
                } else {
                    tile_count = 1;
                    tile_count_locked = 0;
                }
                uart_send_byte(STATUS_OK);
            }
            break;
        case OPC_LOAD_A:
            if (len == 0 || (len % 64) != 0 || len/64 > MAX_TILE_COUNT) { uart_send_byte(STATUS_BADLEN); break; }
            {
                if (stream_len < 8) { uart_send_byte(STATUS_BADLEN); break; }
                int words = len / 4;
                u8 buf[64];
                u32 tile_mat[16];
                u32 packed[8];
                int tile_idx = 0;
                for (int off = 0; off < words; off += 16, tile_idx++) {
                    uart_recv_buf(buf, 64);
                    for (int i = 0; i < 16; i++) {
                        tile_mat[i] = get_u32le(&buf[i*4]);
                        if (off == 0) mat_a[i] = tile_mat[i];
                    }
                    pack_a_words(tile_mat, packed);
                    for (int t = 0; t < stream_len; t++) {
                        u32 v = (t < 8) ? packed[t] : 0;
                        Xil_Out32(BRAM_A_BASE + (tile_idx * stream_len + t)*4, v);
                    }
                }
                // Infer tile_count from payload if not explicitly set >1; otherwise validate.
                int payload_tiles = len / 64;
                if (payload_tiles > 0) {
                    if (tile_count_locked) {
                        if (payload_tiles != tile_count) { uart_send_byte(STATUS_BADLEN); break; }
                    } else {
                        tile_count = (u16)payload_tiles;
                    }
                }
                Xil_DCacheFlushRange(BRAM_A_BASE, tile_count * stream_len * 4);
                uart_send_byte(STATUS_OK);
            }
            break;
        case OPC_LOAD_B:
            if (len == 0 || (len % 64) != 0 || len/64 > MAX_TILE_COUNT) { uart_send_byte(STATUS_BADLEN); break; }
            {
                if (stream_len < 8) { uart_send_byte(STATUS_BADLEN); break; }
                int words = len / 4;
                u8 buf[64];
                u32 tile_mat[16];
                u32 packed[8];
                int tile_idx = 0;
                for (int off = 0; off < words; off += 16, tile_idx++) {
                    uart_recv_buf(buf, 64);
                    for (int i = 0; i < 16; i++) {
                        tile_mat[i] = get_u32le(&buf[i*4]);
                        if (off == 0) mat_b[i] = tile_mat[i];
                    }
                    pack_b_words(tile_mat, packed);
                    for (int t = 0; t < stream_len; t++) {
                        u32 v = (t < 8) ? packed[t] : 0;
                        Xil_Out32(BRAM_B_BASE + (tile_idx * stream_len + t)*4, v);
                    }
                }
                int payload_tiles = len / 64;
                if (payload_tiles > 0) {
                    if (tile_count_locked) {
                        if (payload_tiles != tile_count) { uart_send_byte(STATUS_BADLEN); break; }
                    } else {
                        tile_count = (u16)payload_tiles;
                    }
                }
                Xil_DCacheFlushRange(BRAM_B_BASE, tile_count * stream_len * 4);
                uart_send_byte(STATUS_OK);
            }
            break;
        case OPC_RUN:
            if (len != 0) { uart_send_byte(STATUS_BADLEN); break; }
            {
                // If caller didn't preload BRAM directly, fall back to packing single-tile mats.
                if (tile_count == 1) {
                    u32 a_words[8], b_words[8];
                    pack_a_words(mat_a, a_words);
                    pack_b_words(mat_b, b_words);

                    for (int i = 0; i < 8; i++) {
                        Xil_Out32(BRAM_A_BASE + i*4, a_words[i]);
                        Xil_Out32(BRAM_B_BASE + i*4, b_words[i]);
                    }
                    Xil_DCacheFlushRange(BRAM_A_BASE, sizeof(a_words));
                    Xil_DCacheFlushRange(BRAM_B_BASE, sizeof(b_words));
                }

                write_reg(REG_STREAM_LEN, stream_len);
                write_reg(REG_FLUSH_LEN,  ((u32)tile_count << 16) | (u32)flush_len);
                write_reg(REG_CTRL, 0x2); // clear pulse
                write_reg(REG_CTRL, 0x0);
                write_reg(REG_CTRL, 0x1); // start

                while ((read_reg(REG_STATUS) & 0x1) == 0) {;}
                uart_send_byte(STATUS_OK);
            }
            break;
        case OPC_READ_C:
            if (len != 0) { uart_send_byte(STATUS_BADLEN); break; }
            {
                if (tile_count == 0 || tile_count > MAX_TILE_COUNT) { uart_send_byte(STATUS_BADLEN); break; }
                int total_words = tile_count * 16;
                Xil_DCacheInvalidateRange(BRAM_C_BASE, total_words*4);
                uart_send_byte(STATUS_OK);
                u8 word_bytes[4];
                for (int i = 0; i < total_words; i++) {
                    u32 v = Xil_In32(BRAM_C_BASE + i*4);
                    put_u32le(v, word_bytes);
                    uart_send_buf(word_bytes, 4);
                }
            }
            break;
        case OPC_VERSION:
            if (len != 0) { uart_send_byte(STATUS_BADLEN); break; }
            {
                u8 out[5];
                out[0] = STATUS_OK;
                put_u32le(0x0001u, &out[1]);  // protocol version 1
                uart_send_buf(out, sizeof(out));
            }
            break;
        default:
            uart_send_byte(STATUS_BADOP);
            break;
        }
    }
}
