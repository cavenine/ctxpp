import { readFile } from 'fs/promises';

// Renderable describes something that can be rendered.
interface Renderable {
    render(): string;
}

// WidgetConfig holds widget configuration.
type WidgetConfig = {
    name: string;
    count: number;
};

// Status enumerates widget states.
enum Status {
    Active,
    Inactive,
}

// Widget is a typed UI widget.
class Widget implements Renderable {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    render(): string {
        return `<widget>${this.name}</widget>`;
    }
}

// newWidget creates a Widget.
function newWidget(name: string): Widget {
    return new Widget(name);
}

export { Widget, newWidget };
