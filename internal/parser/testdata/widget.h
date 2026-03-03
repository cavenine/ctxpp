#ifndef WIDGET_H
#define WIDGET_H

#include <stdio.h>
#include <string.h>

/* Widget holds a named UI element. */
typedef struct {
    char name[64];
    int  count;
} Widget;

/* Color enumerates available widget colors. */
typedef enum {
    COLOR_RED,
    COLOR_GREEN,
    COLOR_BLUE
} Color;

/* widget_new initialises a Widget with the given name. */
Widget widget_new(const char *name);

/* widget_render writes the widget's HTML representation. */
void widget_render(const Widget *w, char *buf, size_t buf_len);

/* widget_increment adds n to the widget's internal counter. */
void widget_increment(Widget *w, int n);

/* MAX is a function-like macro. */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#endif /* WIDGET_H */
