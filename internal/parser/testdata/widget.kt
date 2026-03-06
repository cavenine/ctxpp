package demo.ui

import com.example.Renderer
import kotlin.io.println

class Widget(private val count: Int) {
    fun render() {
        println(count)
        helper()
    }

    private fun helper() {}
}

interface Painter {
    fun paint()
}

fun buildWidget(): Widget {
    return Widget(1)
}
