#include <gtk/gtk.h>

// Callback function for the button click event
static void print_hello(GtkWidget *widget, gpointer data) {
    g_print("Hello World\n");
}

// Main function to setup the window and widgets
int main(int argc, char **argv) {
    GtkWidget *window;
    GtkWidget *button;

    gtk_init(&argc, &argv);

    // Create a new window, and set its title
    window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Simple GUI");
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_container_set_border_width(GTK_CONTAINER(window), 10);

    // Create a new button
    button = gtk_button_new_with_label("Click me!");
    g_signal_connect(button, "clicked", G_CALLBACK(print_hello), NULL);

    // Add the button to the window
    gtk_container_add(GTK_CONTAINER(window), button);

    // Show all widgets within the window
    gtk_widget_show_all(window);

    // Run the main loop
    gtk_main();

    return 0;
}
