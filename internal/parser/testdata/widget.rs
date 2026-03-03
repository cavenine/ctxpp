use std::fmt;

/// Widget renders UI elements.
pub struct Widget {
    name: String,
    count: u32,
}

/// Renderer describes anything that can render itself.
pub trait Renderer {
    fn render(&self) -> String;
}

impl Widget {
    /// new creates a Widget with the given name.
    pub fn new(name: &str) -> Self {
        Widget {
            name: name.to_string(),
            count: 0,
        }
    }

    /// increment adds n to the internal counter.
    pub fn increment(&mut self, n: u32) {
        self.count += n;
        self.log(n);
    }

    fn log(&self, n: u32) {
        println!("incremented by {}", n);
    }
}

impl Renderer for Widget {
    fn render(&self) -> String {
        format!("<widget>{}</widget>", self.name)
    }
}

/// Status enumerates widget states.
pub enum Status {
    Active,
    Inactive,
}

/// greet prints a greeting.
pub fn greet(name: &str) {
    println!("hello {}", name);
}
