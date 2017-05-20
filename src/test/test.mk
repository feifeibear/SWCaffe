TEST_DIR=./src/test
$(subst $(TEST_DIR)/%.cpp, $(TEST_DIR)/%.o, ${TEST_OBJ})
