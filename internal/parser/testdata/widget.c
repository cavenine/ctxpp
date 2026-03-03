#include "widget.h"
#include <stdio.h>
#include <string.h>

static void widget_log(const Widget *w, int n);

/* widget_new initialises a Widget with the given name. */
Widget widget_new(const char *name) {
    Widget w;
    strncpy(w.name, name, sizeof(w.name) - 1);
    w.name[sizeof(w.name) - 1] = '\0';
    w.count = 0;
    return w;
}

/* widget_render writes the widget HTML into buf. */
void widget_render(const Widget *w, char *buf, size_t buf_len) {
    snprintf(buf, buf_len, "<widget>%s</widget>", w->name);
}

/* widget_increment adds n to the counter. */
void widget_increment(Widget *w, int n) {
    w->count += n;
    widget_log(w, n);
}

static void widget_log(const Widget *w, int n) {
    printf("widget %s: incremented by %d\n", w->name, n);
}
