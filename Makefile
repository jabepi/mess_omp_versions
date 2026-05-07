CC = g++
CFLAGS = -march=native -Wall -fopenmp
LDFLAGS = -pthread -lrt -fopenmp
#CFLAGS += -no-multibyte-chars added to work around the error Catastrophic error: could not set locale "" to allow processing of multibyte characters

SRCDIR=src
BUILDDIR=build



PRG_SUFFIX=.x
TARGET=stream_omp$(PRG_SUFFIX)

all: init $(TARGET)

init:
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/nop.o: $(SRCDIR)/nop.c
	$(CC) $(CFLAGS) -O3 -c $< -o $@

$(BUILDDIR)/utils.o: $(SRCDIR)/utils.c
	$(CC) $(CFLAGS) -Wno-unused-but-set-variable -O3 -c $< -o $@

$(BUILDDIR)/stream_omp_c.o: $(SRCDIR)/stream_omp.c
	$(CC) $(CFLAGS)  -O3 -c $< -o $@

$(TARGET): $(BUILDDIR)/nop.o $(BUILDDIR)/utils.o $(BUILDDIR)/stream_omp_c.o
	$(CC) $(CFLAGS)  -O3 $(LDFLAGS) $^ -o $@

clean:
	rm -f $(BUILDDIR)/*.o *.x
