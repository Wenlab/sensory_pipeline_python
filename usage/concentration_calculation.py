class ConcentrationCalculator:
    """
    A calculator for determining molar concentrations in aqueous solutions.
    
    Key features:
    - Calculate concentration from molecular weight and dilution
    - Calculate concentration from mass ratio of solute to water
    """
    
    def __init__(self):
        """Initialize the concentration calculator."""
        pass
    
    def concentration_from_dilution(self, molecular_weight, initial_concentration, dilution_factor):
        """
        Calculate molar concentration from molecular weight and dilution.
        
        Args:
            molecular_weight (float): Molecular weight of the compound in g/mol
            initial_concentration (float): Initial concentration in mg/mL or g/L
            dilution_factor (float): Dilution factor (e.g., 1:1000 = 1000)
            
        Returns:
            float: Final molar concentration in µM (micromolar)
        """
        # Convert initial concentration from mg/mL to mM
        initial_molar_concentration_mM = initial_concentration / molecular_weight
        
        # Apply dilution factor
        final_concentration_mM = initial_molar_concentration_mM / dilution_factor
        
        # Convert to µM
        final_concentration_uM = final_concentration_mM * 1000
        
        return final_concentration_uM
    
    def concentration_from_mass_ratio(self, molecular_weight, mass_solute_mg, mass_water_g):
        """
        Calculate molar concentration from mass ratio of solute to water.
        
        Args:
            molecular_weight (float): Molecular weight of the compound in g/mol
            mass_solute_mg (float): Mass of solute in mg
            mass_water_g (float): Mass of water in g
            
        Returns:
            float: Molar concentration in µM (micromolar)
        """
        # Convert solute mass from mg to g
        mass_solute_g = mass_solute_mg / 1000
        
        # Calculate moles of solute
        moles_solute = mass_solute_g / molecular_weight
        
        # Assume density of water is 1 g/mL, so mass in g = volume in mL
        volume_water_mL = mass_water_g
        volume_water_L = volume_water_mL / 1000
        
        # Calculate molarity (mol/L)
        molarity_M = moles_solute / volume_water_L
        
        # Convert to µM
        concentration_uM = molarity_M * 1e6
        
        return concentration_uM
    
    def format_concentration(self, concentration_uM):
        """
        Format concentration with appropriate units and precision.
        
        Args:
            concentration_uM (float): Concentration in µM
            
        Returns:
            str: Formatted concentration string with appropriate units
        """
        if concentration_uM >= 1000:
            return f"{concentration_uM/1000:.2f} mM"
        elif concentration_uM >= 1:
            return f"{concentration_uM:.2f} µM"
        else:
            return f"{concentration_uM*1000:.2f} nM"
    
    def calculate_and_format_from_dilution(self, molecular_weight, initial_concentration, dilution_factor):
        """
        Calculate and format concentration from dilution parameters.
        
        Args:
            molecular_weight (float): Molecular weight in g/mol
            initial_concentration (float): Initial concentration in mg/mL
            dilution_factor (float): Dilution factor
            
        Returns:
            str: Formatted concentration string
        """
        conc_uM = self.concentration_from_dilution(molecular_weight, initial_concentration, dilution_factor)
        return self.format_concentration(conc_uM)
    
    def calculate_and_format_from_mass_ratio(self, molecular_weight, mass_solute_mg, mass_water_g):
        """
        Calculate and format concentration from mass ratio.
        
        Args:
            molecular_weight (float): Molecular weight in g/mol
            mass_solute_mg (float): Mass of solute in mg
            mass_water_g (float): Mass of water in g
            
        Returns:
            str: Formatted concentration string
        """
        conc_uM = self.concentration_from_mass_ratio(molecular_weight, mass_solute_mg, mass_water_g)
        return self.format_concentration(conc_uM)


# Example usage and test cases
if __name__ == "__main__":
    calc = ConcentrationCalculator()
    
    print("=== Concentration Calculator Examples ===\n")
    
    # Example 1: From dilution
    print("Example 1: Dilution method")
    molecular_weight = 180.16  # glucose
    initial_conc = 100  # mg/mL
    dilution = 1000  # 1:1000 dilution
    
    result_uM = calc.concentration_from_dilution(molecular_weight, initial_conc, dilution)
    formatted_result = calc.calculate_and_format_from_dilution(molecular_weight, initial_conc, dilution)
    
    print(f"Molecular weight: {molecular_weight} g/mol")
    print(f"Initial concentration: {initial_conc} mg/mL")
    print(f"Dilution factor: 1:{dilution}")
    print(f"Final concentration: {result_uM:.2f} µM")
    print(f"Formatted: {formatted_result}")
    
    print("\n" + "-"*50 + "\n")
    
    # Example 2: From mass ratio
    print("Example 2: Mass ratio method")
    molecular_weight = 342.3  # sucrose
    mass_solute = 50  # mg
    mass_water = 1000  # g (1 L)
    
    result_uM = calc.concentration_from_mass_ratio(molecular_weight, mass_solute, mass_water)
    formatted_result = calc.calculate_and_format_from_mass_ratio(molecular_weight, mass_solute, mass_water)
    
    print(f"Molecular weight: {molecular_weight} g/mol")
    print(f"Mass of solute: {mass_solute} mg")
    print(f"Mass of water: {mass_water} g")
    print(f"Final concentration: {result_uM:.2f} µM")
    print(f"Formatted: {formatted_result}")