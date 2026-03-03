import { readFile } from 'fs/promises';
import path from 'path';

// Widget renders UI elements.
class Widget {
    constructor(name) {
        this.name = name;
    }

    // render returns the HTML representation.
    render() {
        return `<widget>${this.name}</widget>`;
    }
}

// newWidget creates a new Widget instance.
function newWidget(name) {
    return new Widget(name);
}

// greet sends a greeting.
const greet = (name) => {
    console.log('hello ' + name);
};

export { Widget, newWidget, greet };
