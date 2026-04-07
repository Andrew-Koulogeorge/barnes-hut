CXX = g++
CXXFLAGS = -std=c++17

bhut: bhut.cpp
	$(CXX) $(CXXFLAGS) bhut.cpp -o bhut

clean:
	rm -f bhut