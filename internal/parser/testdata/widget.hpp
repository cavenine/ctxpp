#pragma once
#include <string>

namespace util {

/// Color enumerates widget color options.
enum class Color { Red, Green, Blue };

/// Widget is a typed UI widget.
class Widget {
public:
    /// Widget constructs a widget with the given name.
    explicit Widget(std::string name);

    /// render returns the HTML representation of this widget.
    std::string render() const;

    /// increment adds n to the internal counter.
    void increment(int n);

private:
    std::string name_;
    int count_ = 0;

    void log(int n) const;
};

} // namespace util
