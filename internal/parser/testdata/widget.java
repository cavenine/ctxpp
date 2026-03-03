// Widget is a UI widget.
public class Widget {
    private String name;
    private int count;

    // New creates a Widget with the given name.
    public Widget(String name) {
        this.name = name;
        this.count = 0;
    }

    // render outputs the widget to a stream.
    public String render() {
        return "<widget>" + name + "</widget>";
    }

    // increment adds to the internal count.
    public void increment(int n) {
        this.count += n;
        log(n);
    }

    private void log(int n) {
        System.out.println("incremented by " + n);
    }
}

// Renderer is a rendering interface.
interface Renderer {
    String render();
}
