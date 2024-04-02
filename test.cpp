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

        bool isEmpty() {
            return (top < 0);
        }

        bool isFull() {
            return (top >= max_size - 1);
        }

        void display() {
            if(top < 0) {
                cout << "Stack is empty" << endl;
                return 0;
            }
            for (int i = top; i >= 0; i++) {
                cout << arr[i] << " ";
            }
            cout << endl;
        }

        void delete_specific_element(Stack &stk, int element);
}

// function definition to delete a specific element from the stack
void Stack::delete_specific_element(Stack &stk, int element){
    if(stk.isEmpty()) {
        cout << "Stack is empty" << endl;
        return;
    }

    int size = stk.top + 1;
    int temp[size];
    int count = 0;
    
    // copy stack elements into a temporary array
    while(!stk.isEmpty()) {
        int current = stk.pop();
        if (current != element) {
            temp[count++] = current;
        } 
    }

    // push non matching elements back onto the stack
    for(int i = count - 1; i >= 0; i--) {
        stk.push(temp[i]);
    }

}

int main() {
    
    return 0;
}