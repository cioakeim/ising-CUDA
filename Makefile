
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

v0Test: $(OBJM)/v0Test.o $(OBJ)/isingV0.o 
	@mkdir -p $(BIN)
	$(CC) -o $(BIN)/$@ $^ $(CFLAGS)

$(OBJM)/%.o: $(MAIN)/%.cu 
	@mkdir -p $(OBJM)
	$(CC) -c $< -o $@ $(CFLAGS)  

$(OBJ)/isingV0.o: $(SRC)/isingV0.cu 
	@mkdir -p $(OBJ)
	$(CC) -c $< -o $@ $(CFLAGS)  

$(OBJ)/%.o: $(SRC)/%.cu 
	@mkdir -p $(OBJ)
	$(CC) -c $< -o $@ $(CFLAGS)  

clean:
	rm -rf $(OBJ)/* $(OBJM)/* $(BIN)/*
