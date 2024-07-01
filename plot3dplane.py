"""
This script provides an interactive Matplotlib window to experiment with 
the coefficients and constants of the General Form equation of a plane. 
The plotted plane updates instantly when the coefficients or constants 
are changed.

Feel free to use use any of the classes from this script in your own 
projects.

To Run the script:
    $ python plotplane3d.py
"""

# Standard libraries
import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import TypeAlias, Any

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
    """Represents the equation of a plane in General Form.
    
    Args:
        a: The coefficient a. Default 0.0
        b: The coefficient b. Default 0.0
        c: The coefficient c. Must be more than zero. Default 0.01
        known_point: A known point on the plane. Default Point(x=0.0, y=0.0, z=0.0)

    Note: 
        The General Form Equation of a plane is:
        
            a(x - x1) + b(y - y1) + c(z - z1) = 0

        - where:
            - a, b, c are coefficients, with c never equal to zero
            - x1, y1, z1 represent a known point on the plane
            - x, y, z represents any unknown point on the plane
    
    Raises:
        ValueError: Coefficient c cannot be within 1e-9 of zero.
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
    """Represents the equation of a plane in Computational Form.
    
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
        instance of PlaneGeneralForm.

        Args:
            gen_equation: The general form of the plane equation.

        Note:
            - This is an alternative constructor that accepts an instance of 
            PlaneGeneralForm and then transforms it into an instance of 
            PlaneComputationalForm.
        """
        a = -1.0 * (gen_equation.a / gen_equation.c)
        b = -1.0 * (gen_equation.b / gen_equation.c)
        addend0 = (gen_equation.a * gen_equation.x1) / gen_equation.c
        addend1 = (gen_equation.b * gen_equation.y1) / gen_equation.c
        addend2 = gen_equation.z1
        c = addend0 + addend1 + addend2

        return cls(a, b, c)
    
    @staticmethod
    def generate_xy_pairs(axis_draw_length: int = 20) -> XYPairs:
        """Generate 400 X, Y pairs (samples) that can be used to calculate Z 
        values.

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

        # Fixed number of points per axis for consistent performance
        points_per_axis = 20

        x = np.linspace(low, high, points_per_axis)
        y = np.linspace(low, high, points_per_axis)
        X, Y = np.meshgrid(x, y)

        return X, Y

    def calculate_z_values(self, mesh_grid: XYPairs) -> npt.NDArray[np.float64]:
        """Calculate Z values from meshgrid.
        
        Args:
            mesh_grid: The x, y value pairs used to calculate Z.

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
    """The window that displays the plotted plane in 3D space.

    The `PlottingWindow` class is designed to only allow for a single instance
    of a plotting window. 

    Example::

        plane = PlaneGeneralForm(0.0, 0.0, 0.001, Point(0, 0, 0))
        plotting_window = PlottingWindow(plane)
    """

    # Declare the text box attributes
    text_box_a: TextBox
    text_box_b: TextBox
    text_box_c: TextBox
    text_box_x1: TextBox
    text_box_y1: TextBox
    text_box_z1: TextBox

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
        
        PlottingWindow._initialized = True

        self.__initialize_plot()
        self.__initialize_text_boxes()

        # Set up the event handlers for the text boxes
        self.text_box_a.on_text_change(self.__update)
        self.text_box_b.on_text_change(self.__update)
        self.text_box_c.on_text_change(self.__update)
        self.text_box_x1.on_text_change(self.__update)
        self.text_box_y1.on_text_change(self.__update)
        self.text_box_z1.on_text_change(self.__update)

        x, y, z = self.calculate_xyz_table_for_plane(
            plane=PlaneComputationalForm.init_from_general_form(self._plane_general_form),
            known_point=self._plane_general_form.known_point
        )
        self.__plot_plane_surface(x, y, z)
        self.__plot_known_point(self._plane_general_form.known_point)
        self._ax.legend(loc='best')
        plt.show()

    def __initialize_plot(self):
        """Initialize the matplotlib plot.

        Sets up the figure and axis for a 3D plot. Adjusts the subplot to prevent
        overlap with widgets and sets up the 3D projection.
        """
        self.fig, self._ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)  # Adjust to prevent overlap of widgets and plot
        self._ax = self.fig.add_subplot(111, projection='3d')

    @unique
    class __BoxXLocal(Enum):
        """The predefined x locations a texbox could be located within the 
        matplotlib UI window.
        
        Note:
            Each column is a location along the x-axis of the matplotlib UI widget.
        """
        
        COLUMN0 = 0.3
        COLUMN1 = 0.5
        COLUMN2 = 0.7

    @unique
    class __BoxYLocal(Enum):
        """The predefined y locations a texbox could be located within the 
        matplotlib UI window.
        
        Note:
            Each row is a location along the y-axis of the matplotlib UI widget.
        """
        ROW0 = 0.05
        ROW1 = 0.15

    @dataclass
    class __TextBoxShape:
        """The location and dimensions of a textbox UI element located within 
        the matplotlib UI window."""

        x_location: 'PlottingWindow.__BoxXLocal'
        y_location: 'PlottingWindow.__BoxYLocal'
        width: float = 0.1
        height: float = 0.05

        @property
        def as_tuple(self) -> tuple[float, float, float, float]:
            return self.x_location.value, self.y_location.value, self.width, self.height

    def __initialize_text_boxes(self):
        """Initialize TextBoxes so the user can modify the plotted plane.

        Creates TextBoxes for each coefficient and known point coordinate,
        setting up their initial values based on the plane's attributes.
        """
        
        text_box_configs = {
            'a': ('Plane Coefficients ☞     $a$ ', self._plane_general_form.a, 
                  PlottingWindow.__BoxXLocal.COLUMN0, PlottingWindow.__BoxYLocal.ROW0),
            'b': ('$b$ ', self._plane_general_form.b, 
                  PlottingWindow.__BoxXLocal.COLUMN1, PlottingWindow.__BoxYLocal.ROW0),
            'c': ('$c$ ', self._plane_general_form.c, 
                  PlottingWindow.__BoxXLocal.COLUMN2, PlottingWindow.__BoxYLocal.ROW0),
            'x1': ('Known Point in 3D space ☞   $x1$ ', self._plane_general_form.x1, 
                   PlottingWindow.__BoxXLocal.COLUMN0, PlottingWindow.__BoxYLocal.ROW1),
            'y1': ('$y1$ ', self._plane_general_form.y1, 
                   PlottingWindow.__BoxXLocal.COLUMN1, PlottingWindow.__BoxYLocal.ROW1),
            'z1': ('$z1$ ', self._plane_general_form.z1, 
                   PlottingWindow.__BoxXLocal.COLUMN2, PlottingWindow.__BoxYLocal.ROW1)
        }
        for key, (label, initial, x_loc, y_loc) in text_box_configs.items():
            txtbox_loc = PlottingWindow.__TextBoxShape(
                x_location=x_loc,
                y_location=y_loc
            )
            setattr(self, f'text_box_{key}', TextBox(
                ax=plt.axes(txtbox_loc.as_tuple),
                label=label,
                initial=str(initial)
            ))

    def __update(self, val: Any):
        """Update the plot with the newly changed values from the text boxes.
        
        When the user changes the values in the text boxes, this method is called
        to update the plane's equation and replot the plane with the new values.
        """
        try:
            a = float(self.text_box_a.text)
            b = float(self.text_box_b.text)
            c = float(self.text_box_c.text)
            x1 = float(self.text_box_x1.text)
            y1 = float(self.text_box_y1.text)
            z1 = float(self.text_box_z1.text)
        except ValueError:
            # NOTE: Plane will only update when the textbox contains a numeric value
            pass
        else:
            new_known_point = Point(x1, y1, z1)

            c = c if c != 0 else 0.001  # Prevent divide by zero error
            new_plane = PlaneGeneralForm(a, b, c, known_point=new_known_point)

            x, y, z = self.calculate_xyz_table_for_plane(
                plane=PlaneComputationalForm.init_from_general_form(new_plane),
                known_point=new_known_point
            )
            self._ax.clear()  # Clear the axis for the new plot
            self.__plot_plane_surface(x, y, z)
            self.__plot_known_point(new_known_point)
            plt.draw()

    def __calculate_draw_dimensions(self, known_point: Point) -> int:
        """Calculate the drawing dimensions to ensure the known point
        appears correctly on the plane.

        This method adjusts the drawing dimensions so that the dot
        representing the known point on the plane doesn't appear
        to be off the plane. This can happen if the plane is drawn
        too small to fit the dot.

        The integer returned represents the range of values for both x and y, 
        so a value of 20 would mean −10≤x≤10 and −10≤y≤10
        """
        draw_dimensions = 20  # Smallest possible size.
        x_coordinate = abs(round(known_point.x))
        y_coordinate = abs(round(known_point.y))
        if x_coordinate > (draw_dimensions // 2):
            draw_dimensions = round(x_coordinate * 0.25) + (x_coordinate * 2)
        if y_coordinate > (draw_dimensions // 2):
            draw_dimensions = round(y_coordinate * 0.25) + (y_coordinate * 2)
        if not (draw_dimensions % 2 == 0):
            draw_dimensions += 1
        return draw_dimensions

    def calculate_xyz_table_for_plane(
            self, 
            plane: PlaneComputationalForm,
            known_point: Point 
            ) -> tuple[
                npt.NDArray[np.float64], 
                npt.NDArray[np.float64], 
                npt.NDArray[np.float64]
                ]:
        """Calculate the xyz table for a given plane equation. The table is 
        required to properly plot the plane.
        
        Args:
            plane: The equation of a plane in computational form
            known_point: A point that is to be highlighted on the plane surface

        Returns:
            A tuple containing three Numpy Arrays in the form X, Y and Z. The 
            first column is X where the last is Z
        """
        # Calculate the xyz values that are required to plot a plane in 3D
        plot_range = self.__calculate_draw_dimensions(known_point)
        mesh_grid = plane.generate_xy_pairs(axis_draw_length=plot_range)
        Z = plane.calculate_z_values(mesh_grid)
        X, Y = mesh_grid
        return X, Y, Z

    def __plot_plane_surface(
            self, 
            x: npt.NDArray[np.float64], 
            y: npt.NDArray[np.float64], 
            z: npt.NDArray[np.float64]
        ):
        """Plots a surface when provided with the x, y, z values 
        that make up a table of values to be plotted.

        Configures the axis labels and title before plotting the surface.
        """
        self._ax.set_xlabel('X coordinates')
        self._ax.set_ylabel('Y coordinates')
        self._ax.set_zlabel('Z coordinates')
        title = '3D Plane from equation $a(x - x1) + b(y - y1) + c(z - z1) = 0$'
        self._ax.set_title(title)
        self._ax.plot_surface(x, y, z, alpha=0.5, rstride=1, cstride=1, color='b', label='Plane Surface')

    def __plot_known_point(self, known_point: Point):
        """Plot the point from the general equation and highlight it as orange. 

        Args:
            known_point: The known point on the plane that's to be plotted.
        """
        self._ax.scatter(
            known_point.x,
            known_point.y,
            known_point.z,
            color='orange',
            s=100,
            label='Known Point on Plane'
        )


if __name__ == '__main__':
    plane = PlaneGeneralForm(0.0, 0.0, 0.001, Point(0, 0, 0))
    PlottingWindow(plane)
