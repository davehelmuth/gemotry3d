"""
This script defines and demonstrates the use of classes for representing and 
manipulating a plane in 3D space, utilizing both general and computational 
forms of the plane's equation.

"""

# Standard libraries
import math
from dataclasses import dataclass, field
from typing import TypeAlias

# Installed libraries
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox


@dataclass
class Point:
    """A point in 3D space
    
    Args:
        x: The x coordinate of the point. Defaults to 0.0.
        y: The y coordinate of the point. Defaults to 0.0.
        z: The z coordinate of the point. Defaults to 0.0.
    """

    x: float = field(default=0.0)
    """The x coordinate of the point"""

    y: float = field(default=0.0)
    """The y coordinate of the point"""

    z: float = field(default=0.0)
    """The z coordinate of the point"""


@dataclass
class PlaneGeneralForm:
    """Represents the equation of plane in General Form.
    
    Args:
        a: The coefficient a. Default 0.0
        b: The coefficient b. Default 0.0
        c: The coefficient c. Must be more than zero  Default 0.01
        known_point: A known point on the plane. Default Point(x=0.0, y=0.0, z=0.0)

    Note: 
        The General Form Equation of a plane is:
        
            a(x - x1) + b(y - y1) + c(z - z1) = 0

        - where:
            - a, b, c are coefficients of which c is never equal to zero
            - x1, y1, z1 represent a known point on the plane
            - x, y, z represents any unknown point on the plane
    
    Raises:
        ValueError: Coefficient c can not be within 1e-9 of zero
    """

    a: float = field(default=0.0)
    """The coefficient a"""

    b: float = field(default=0.0)
    """The coefficient b"""

    c: float = field(default=0.01)
    """The coefficient c"""

    known_point: Point = field(default_factory=lambda: Point(x=0.0, y=0.0, z=0.0))
    """A known point on the plane."""

    def __post_init__(self):
        if math.isclose(self.c, 0.0, abs_tol=1e-9):
            raise ValueError("Coefficient 'c' cannot be close to zero.")

    @property
    def x1(self) -> float:
        """The x-coordinate from the known point on the plane."""
        return self.known_point.x
    
    @property
    def y1(self) -> float:
        """The y-coordinate from the known point on the plane."""
        return self.known_point.y

    @property
    def z1(self) -> float:
        """The z-coordinate from the known point on the plane."""

        return self.known_point.z


XYPairs: TypeAlias = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
"""
A tuple of two 2D NumPy arrays representing the X and Y coordinate pairs 
that make up a mesh grid.

This type alias is used to represent a pair of 2D NumPy arrays where:
- The first element in the Tuple is the X coordinates array.
- The second element in the Tuple is the Y coordinates array.

The arrays are typically generated using `numpy.meshgrid` from two 1D arrays 
created by `numpy.linspace`. These pairs of X and Y coordinates can be used 
for further calculations, such as evaluating a function Z = f(X, Y) over a 
grid in a 3D space.

Examples::

    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)
    xy_pair: XYPairs = (X, Y)
"""


class PlaneComputationalForm:
    """Represents the equation of plane in Computational Form.
    
    To be utilized by software, the equation of a plane must be in this form. 

    Note:
        The equation in computational form is:

            Z = aX + bY + c

        - where:
            - `a`, `b` are coefficients and `c` is a constant
                - These are not the same values from the general equation.
            - X, Y each pair calculates a unique value of Z 

    """

    def __init__(self, a: float, b: float, c: float) -> None:
        """
        Args:
            a: The coefficient `a` in equation `Z = aX + bY + c`
            b: The coefficient `b` in equation `Z = aX + bY + c`
            c: The constant `c` in equation `Z = aX + bY + c`
        """
        self.a = a
        self.b = b
        self.c = c

    @classmethod
    def init_from_general_form(cls, gen_equation: PlaneGeneralForm) -> 'PlaneComputationalForm':
        """Initialize an instance of PlaneComputationalForm by transforming an 
        instance PlaneGeneralForm.

        Args:
            gen_equation: The general form of the plane equation.

        Note:
            - This is an alternative constructor when you want to convert an 
            instance of PlaneGeneralForm to the computation form. 
        """

        a = -1.0 * (gen_equation.a / gen_equation.c)
        b = -1.0 * (gen_equation.b / gen_equation.c)

        addend0 = (gen_equation.a * gen_equation.x1) / gen_equation.c
        addend1 = (gen_equation.b * gen_equation.y1) / gen_equation.c
        addend2 = gen_equation.z1
        c = addend0 + addend1 + addend2

        return cls(a, b, c)
    
    @staticmethod
    def generate_xy_pairs(axis_draw_length: int=20) -> XYPairs:
        """Generate a minimum of 400 X, Y pairs that can be used to 
        calculate z values.

        Args:
            axis_draw_length: The size of both the x and y axis.

        Returns:
            A mesh-like 2D grid of x, y pairs to be used to calculate the Z 
            values for a function (i.e., plane) in 3D space.

        Raises:
            ValueError: If axis_draw_length is less than 20.

        Notes:
            The axis limits are symmetrically centered around zero. The limits 
            are determined by dividing the axis_draw_length by 2 and rounding 
            to the nearest integer.
        """

        if axis_draw_length < 20:
            raise ValueError("Minimum valid value is 20")

        low = 0 - axis_draw_length // 2
        high = 0 + axis_draw_length // 2

        x = np.linspace(low, high, axis_draw_length)
        y = np.linspace(low, high, axis_draw_length)
        X, Y = np.meshgrid(x, y)

        return X, Y

    def calculate_z_values(self, mesh_grid: XYPairs) -> npt.NDArray[np.float64]:
        """Calculate Z Values from meshgrid.
        
        Args:
            mesh_grid: The x,y value pairs used to calculate z

        Returns:
            np.ndarray: Computed Z values.
        """

        X, Y = mesh_grid

        # Computationally friendly version of plane equation
        Z = self.a * X + self.b * Y + self.c

        return Z

    def __repr__(self) -> str:
        return f'PlaneComputationalForm(a={self.a}, b={self.b}, c={self.c})'



class PlottingWindow:
    """The Window that displays the plotted plane in 3D space.

    The `PlottingWindow` class is designed to only allow for a single instance
    of a plotting window. 

    Example::

        plane = PlaneGeneralForm(0.0, 0.0, 0.001, Point(0, 0, 0))
        plotting_window = PlottingWindow(plane)

    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PlottingWindow, cls).__new__(cls)
        return cls._instance

    def __init__(self, plane: PlaneGeneralForm):
        """Provide the plane you would like to plot in 3D space.

        Args:
            plane: The plane object to be plotted.
        
        """
        if PlottingWindow._initialized:
            return
        
        self._plane_general_form = plane
        
        self.initialized = True

        self.__setup_plot()

        self.__initialize_text_boxes()

        # Set up the event handlers for the text boxes
        self.text_box_a.on_text_change(self.__update)
        self.text_box_b.on_text_change(self.__update)
        self.text_box_c.on_text_change(self.__update)
        self.text_box_x1.on_text_change(self.__update)
        self.text_box_y1.on_text_change(self.__update)
        self.text_box_z1.on_text_change(self.__update)

        self.__plot_plane(PlaneComputationalForm.init_from_general_form(plane))
        self.__show()


    def __setup_plot(self):
        self.fig, self._ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)  # Adjust to prevent overlap of widgets and plot
        self._ax = self.fig.add_subplot(111, projection='3d')


    def __initialize_text_boxes(self):
        # Initialize TextBoxes for user to modify the plotted plane

        self.text_box_a = TextBox(
            plt.axes((0.2, 0.05, 0.1, 0.05)), '$a$ ', initial=str(plane.a)
        )
        self.text_box_b = TextBox(
            plt.axes((0.4, 0.05, 0.1, 0.05)), '$b$ ', initial=str(plane.b)
        )
        self.text_box_c = TextBox(
            plt.axes((0.6, 0.05, 0.1, 0.05)), '$c$ ', initial=str(plane.c)
        )
        self.text_box_x1 = TextBox(
            plt.axes((0.2, 0.15, 0.1, 0.05)), '$x1$ ', initial=str(plane.x1)
        )
        self.text_box_y1 = TextBox(
            plt.axes((0.4, 0.15, 0.1, 0.05)), '$y1$ ', initial=str(plane.y1)
        )
        self.text_box_z1 = TextBox(
            plt.axes((0.6, 0.15, 0.1, 0.05)), '$z1$ ', initial=str(plane.z1)
        )

    def __update(self, val):

        try: 
            a = float(self.text_box_a.text)
            b = float(self.text_box_b.text)
            c = float(self.text_box_c.text)
            x1 = float(self.text_box_x1.text)
            y1 = float(self.text_box_y1.text)
            z1 = float(self.text_box_z1.text)


            known_point = Point(x=x1, y=y1, z=z1)
            updated_plane = PlaneGeneralForm(a=a, b=b, c=c, known_point=known_point)
            self._plane_general_form = updated_plane  # A dependency for __plot_plane
            computational_plane = PlaneComputationalForm.init_from_general_form(updated_plane)

            self._ax.clear()  # Clear the axis for the new plot
            self.__plot_plane(computational_plane)
            plt.draw()
        except Exception:
            pass

    def __plot_plane(self, plane: PlaneComputationalForm):
        known_point = self._plane_general_form.known_point

        draw_dimensions = 20  
        x_coordinate = abs(round(known_point.x))
        y_coordinate = abs(round(known_point.y))
        if x_coordinate > (draw_dimensions // 2):
            draw_dimensions = round(x_coordinate * 0.25) + (x_coordinate * 2)
        if y_coordinate > (draw_dimensions // 2):
            draw_dimensions = round(y_coordinate * 0.25) + (y_coordinate * 2)
        if not (draw_dimensions % 2 == 0):
            draw_dimensions = draw_dimensions + 1

        mesh_grid = plane.generate_xy_pairs(axis_draw_length=draw_dimensions)
        Z = plane.calculate_z_values(mesh_grid)
        X, Y = mesh_grid
        self._ax.plot_surface(X, Y, Z, alpha=0.5, rstride=1, cstride=1, color='b')
        self._ax.scatter(
            known_point.x,
            known_point.y, 
            known_point.z, 
            color='orange', 
            s=100, 
            label='Known Point on Plane'
        )
        self._ax.set_xlabel('X coordinates')
        self._ax.set_ylabel('Y coordinates')
        self._ax.set_zlabel('Z coordinates')
        title = '3D Plane from equation $a(x - x1) + b(y - y1) + c(z - z1) = 0$'
        self._ax.set_title(title)

    def __show(self):
        plt.show()


# Usage example:
if __name__ == '__main__':
    plane = PlaneGeneralForm(0.0, 0.0, 0.001, Point(0, 0, 0))
    plotting_window = PlottingWindow(plane)
