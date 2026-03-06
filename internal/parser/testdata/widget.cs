using System;
using Demo.Rendering;

namespace Demo.App;

public class Widget {
    private int count;

    public void Render() {
        Console.WriteLine(count);
        Helper();
    }

    private void Helper() {}
}

public interface IRenderer {
    void Draw();
}
