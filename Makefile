CC = g++
CFLAGS = -Wall -O3
TARGET = src/cpp/radio_engine

all: $(TARGET)

$(TARGET): src/cpp/radio_channel.cpp
	$(CC) $(CFLAGS) src/cpp/radio_channel.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)