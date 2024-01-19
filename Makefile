
SRC=src
OBJ=obj
MAIN=main
OBJM=objmain
INC=include
BIN=bin

CC=nvcc
CFLAGS=-Iinclude

.PHONY: all clean

v0visual: $(MAIN)/v0visual.c
	mkdir -p $(BIN)
	gcc -o $(BIN)/$@ $^

v0test: $(OBJM)/v0Test.o $(OBJ)/isingV0.o 
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v0time: $(OBJM)/v0time.o $(OBJ)/isingV0.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v1test: $(OBJM)/v1test.o $(OBJ)/isingV0.o $(OBJ)/isingV1.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v1time: $(OBJM)/v1time.o $(OBJ)/isingV1.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

$(OBJM)/%.o: $(MAIN)/%.cu 
	@mkdir -p $(OBJM)
	$(CC) -c $< -o $@ $(CFLAGS)  

$(OBJ)/isingV0.o: $(SRC)/isingV0.cu 
	@mkdir -p $(OBJ)
	$(CC) -c $< -o $@ $(CFLAGS)  

$(OBJ)/isingV1.o: $(SRC)/isingV1.cu
	@mkdir -p $(OBJ)
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -rf $(OBJ)/* $(OBJM)/* $(BIN)/*
