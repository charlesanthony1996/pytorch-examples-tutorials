// linked list operations with memory allocation

#include <iostream>
#include <string>


struct Node {
    int data;
    Node * next;
};


// function to display the list
void displayList(Node * head) {
    Node * current = head;
    while(current != nullptr) {
        std::cout << current -> data << " ";
        current = current -> next;
    }
    std::cout << std::endl;
}

// function to insert a node at the beginning of the linked list
void insertAtBeginning(Node * & head, int value) {
    Node * newNode = new Node;
    newNode -> data = value;
    newNode -> next = head;
    head = newNode;
}

// function to delete a node from the linked list
void deleteNode(Node * & head, int value) {
    if(head == nullptr) {
        // empty list nothing to delete
        return;
    }

    if(head -> data == value) {
        // store the current head in a temporary node 
        Node * temp = head;
        // move the head to the next node
        head = head -> next;
        // delete the original head node
        delete temp;
        return;
    }
    // init a pointer to head
    Node * current = head;
    // traverse until the end of the list
    while(current-> next-> data == value) {
        // store the next node in a temporary node
        Node * temp = current -> next;
        // update the pointers to skip the node to be deleted
        current -> next = temp -> next;
        // delete the node with the value
        delete temp;
        return;
    }
    // move to the next node
    current = current -> next;
}

// function to deallocate the memory of the linked list
void deleteList(Node * & head) {
    Node * current = head;
    while(current != nullptr) {
        Node * temp = current;
        current = current -> next;
        delete temp;
    }
    head = nullptr;
}




int main() {
    Node * head = nullptr;

    // insert
    insertAtBeginning(head, 1);
    insertAtBeginning(head, 3);
    insertAtBeginning(head, 5);
    insertAtBeginning(head, 13);

    // display the list
    std::cout << "Initial list: ";
    displayList(head);



    
    return 0;
}