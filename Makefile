
SRC=src
OBJ=obj
MAIN=main
OBJM=objmain
INC=include
BIN=bin

CC=nvcc
CFLAGS=-Iinclude

.PHONY: all clean

list:
	@echo $(SRC)
	@echo $(INC)
	@echo $(OBJ)
	@ls $(SRC)/*.cu
	@ls $(INC)/*.h

all: v0visual v0test v0time v1test v1time v2test v2time v3test v3time

v0visual: $(MAIN)/v0visual.c 
	mkdir -p $(BIN)
	gcc -o $(BIN)/$@ $^

v0test: $(OBJM)/v0test.o $(OBJ)/isingV0.o 
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

v2test: $(OBJM)/v2test.o $(OBJ)/isingV1.o $(OBJ)/isingV2.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v2time: $(OBJM)/v2time.o $(OBJ)/isingV2.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v3test: $(OBJM)/v3test.o $(OBJ)/isingV2.o $(OBJ)/isingV3.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v3time: $(OBJM)/v3time.o $(OBJ)/isingV3.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

v4test: $(OBJM)/v4test.o $(OBJ)/isingV2.o $(OBJ)/isingV3.o
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

$(OBJM)/%.o: $(MAIN)/%.cu 
	@mkdir -p $(OBJM)
	$(CC) -c $< -o $@ $(CFLAGS)  

$(OBJ)/%.o: $(SRC)/%.cu 
	@mkdir -p $(OBJ)
	$(CC) -c $< -o $@ $(CFLAGS)  

$(SRC)/%.cu: $(INC)/%.h
	touch $@

clean:
	rm -rf $(OBJ)/* $(OBJM)/* $(BIN)/*
