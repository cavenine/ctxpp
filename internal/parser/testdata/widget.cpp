#include "widget.hpp"
#include <sstream>
#include <iostream>

namespace util {

Widget::Widget(std::string name) : name_(std::move(name)), count_(0) {}

std::string Widget::render() const {
    std::ostringstream oss;
    oss << "<widget>" << name_ << "</widget>";
    return oss.str();
}

void Widget::increment(int n) {
    count_ += n;
    log(n);
}

void Widget::log(int n) const {
    std::cout << "widget " << name_ << ": incremented by " << n << "\n";
}

/// square returns the square of x.
template <typename T>
T square(T x) {
    return x * x;
}

} // namespace util
