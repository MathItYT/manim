# Manim Web: The ManimCE fork to create interactive math animations in the web
Do you want to turn your ManimCE code into an interactive web animation? This is your library!

## Main changes
* **Asynchronous animations:** [`self.play`](http://self.play) and `self.wait` now must be awaited, so `self.construct` method is also an asynchronous function.
* **MathJax LaTeX rendering:** As we're using a web browser, we can't use system's LaTeX. Instead, we use a really faster implementation called MathJax, that delivers math equations for the web. Notice that there's no `Tex`, so you can do `MathTex(f"\\text{{{tex_code}}}")` to render LaTeX text instead of `Tex(tex_code)`.
* **Text rendering without Pango (under development):** As Pango needs a system, we can't render text with Pango, but we'll use JavaScript libraries to handle that stuff (you don't need any JS, just internal working). This is still under development, so if you try to call `Text("Hello, world!")`, this will raise an error, because `Text` isn't still ready!
* **Interactive animations and integration with JavaScript:** You can now create interactive animations that can be controlled by the user. This is done by using JavaScript to handle events and Manim Web to create the animations.

## Installation
Wait, this is not a system library, so you don't need to install it! In fact, you must start with an HTML file that includes [Pyodide](https://pyodide.org/en/stable/) and loads Manim Web with `micropip.install("manim-web")`. An example is at the demo section below.

## Demo
You have an example at [https://mathityt.github.io/manim-web-demo/](https://mathityt.github.io/manim-web-demo/). The source code is at [https://github.com/MathItYT/manim-web-demo](https://github.com/MathItYT/manim-web-demo/blob/main/index.html).

## Usage
You can use Manim Web like you use ManimCE, but with some differences. Notice that many stuff are asynchronous (like `self.construct`, `self.play`, `self.wait`, etc.), so you must use `async/await` syntax. Here's an example of a simple scene:

```python
from manim import *


class MyScene(Scene):
    async def construct(self):
        # Create a square
        square = Square()
        # Add the square to the scene
        self.add(square)
        # Wait for 1 second
        await self.wait(1)
        # Rotate the square
        await self.play(square.animate.rotate(PI / 4))
        # Wait for 1 second
        await self.wait(1)
```

## Limitations
* **No Pango text rendering:** As mentioned above, Pango is not available in the web, so text rendering is not available yet. But `Text` will be available soon!
* **Retro-compatibility:** As mentioned above, many methods are asynchronous, so you must use `async/await` syntax. This means that you must replace all `self.play` and `self.wait` calls with `await self.play` and `await self.wait`, and also make the `construct` method asynchronous by using `async def construct(self)`.

## Acknowledgements
Thanks to the Manim Community developers and 3Blue1Brown for developing a great animation engine!
