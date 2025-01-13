import csv
from dataclasses import dataclass
from typing import Tuple, Optional
from pint import UnitRegistry

# Initialize Pint unit registry
ureg = UnitRegistry()
Q_ = ureg.Quantity

@dataclass
class Material:
    name: str
    young_modulus: Q_  # type: ignore # Young's modulus in Pascals
    young_modulus_unit: str = "Pa"
    poisson_ratio: float  # Poisson's ratio (unitless)
    density: Q_  # type: ignore # Density in kg/m³
    density_unit: str = "kg/m³"
    color: Tuple[int, int, int]  # RGB color as a tuple
    hardness: Optional[float] = None  # Optional hardness value (unitless)

materials = []

# Path to your external CSV file
file_path = "materials.csv"  # Replace with the actual path to your CSV file

# Open the CSV file and parse its content
with open(file_path, newline='', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        # Parse values and ensure all units are in Pascals and kg/m³
        young_modulus = Q_(float(row["young_modulus"]), ureg.pascal)
        density = Q_(float(row["density"]), ureg.kilogram / ureg.meter**3)

        material = Material(
            name=row["name"],
            young_modulus=young_modulus.magnitude,
            young_modulus_unit=str(young_modulus.units),
            poisson_ratio=float(row["poisson_ratio"]),
            density=density.magnitude,
            density_unit=str(density.units),
            color=(
                int(row["color_r"]),
                int(row["color_g"]),
                int(row["color_b"])
            ),
            hardness=float(row["hardness"]) if row["hardness"] else None
        )
        materials.append(material)

# Example: Print material data in Pascals and kg/m³
for mat in materials:
    print(f"{mat.name}: Young's Modulus = {mat.young_modulus} {mat.young_modulus_unit}, "
          f"Density = {mat.density} {mat.density_unit}")

