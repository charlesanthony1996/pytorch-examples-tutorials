// delete all occurences of an item in a stack
#include <algorithm>
#include <deque>
#include <iostream>

using namespace std;

class Stack {
    private:
        int top;
        int arr[max_size];

    public:
        Stack() {
            top = -1;
        }

        bool push(int x) {
            if (isFull()) {
                cout << "Stack overflow" << endl;
                return false;
            }

            arr[++top] = x;
            return true;
        }

        int pop() {
            if(isEmpty()) {
                cout << "Stack is empty: " << endl;
                return 0;
            }
            return arr[top--];
        }

        int peek() {
            if(isEmpty()) {
                cout << "Stack is empty: " << endl;
                return 0;
            }
            return arr[top];
        }
}

int main() {
    return 0;
}