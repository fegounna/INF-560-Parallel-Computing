SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=gcc
NVCC=nvcc
CFLAGS=-O3 -I$(HEADER_DIR) -fopenmp
NVCCFLAGS=-O3 -I$(HEADER_DIR) -Xcompiler -fopenmp

SRC_C= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	main.c \
	openbsd-reallocarray.c \
	quantize.c

SRC_CU= blur_filter.cu \
	sobel_filter.cu \
	sobel_filter_tiled.cu  

OBJ_C=$(patsubst %.c, $(OBJ_DIR)/%.o, $(SRC_C))
OBJ_CU=$(patsubst %.cu, $(OBJ_DIR)/%.o, $(SRC_CU))

OBJ=$(OBJ_C) $(OBJ_CU)

all: $(OBJ_DIR) sobelf

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile C files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $(SRC_DIR)/$*.c

# Compile CUDA files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $(SRC_DIR)/$*.cu

# Linking
sobelf: $(OBJ)
	$(NVCC) -o $@ $^ -Xcompiler -fopenmp

clean:
	rm -f sobelf $(OBJ)
