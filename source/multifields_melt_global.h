/*
  This file is adapted from melt_global.h/cc, the original file can
  be found in include/material_model/melt_global.h

  Like other files in ASPECT, this file is open to anyone who is interested. 

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_material_model_multifields_melt_global_h
#define _aspect_material_model_multifields_melt_global_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator.h>
#include <aspect/simulator_access.h>
#include <aspect/postprocess/melt_statistics.h>
#include <aspect/melt.h>

namespace aspect
{
	namespace MaterialModel
	{
		using namespace dealii;

		/**
         * A material model that implements a simple formulation of the
         * material parameters required for the modelling of melt transport
         * in a global model, including a source term for the porosity according
         * a simplified linear melting model.
         *
         * The model is considered incompressible, following the definition
         * described in Interface::is_compressible.
         *
         * The theory of this model is based on
         * Juliane Dannberg, Timo Heister, Compressible magma/mantle dynamics: 3-D, adaptive simulations in ASPECT, Geophysical Journal International, 2016
         * @ingroup MaterialModels
         */
		template <int dim>
		class MultiMeltGlobal : public MaterialModel::MeltInterface<dim>, public ::aspect::SimulatorAccess<dim>, public MaterialModel::MeltFractionModel<dim>
		{
		public:

			/**
			 * Return whether the model is compressible or not.
			 */
			bool is_compressible() const override;

			/**
			 * Return the reference viscosity.
			 */
			double reference_viscosity() const override;

			double reference_darcy_coefficient() const override;

			/**
             * Compute the equilibrium melt fractions for the given input conditions.
             * @p in and @p melt_fractions need to have the same size.
             *
             * @param in Object that contains the current conditions.
             * @param melt_fractions Vector of doubles that is filled with the
             * equilibrium melt fraction for each given input conditions.
             */
			void evaluate(const typename Interface<dim>::MaterialModelInputs &in, typename Interface<dim>::MaterialModelOutputs &out) const override;

			void melt_fractions(const MaterialModel::MaterialModelInputs<dim> &in, std::vector<double> &melt_fractions) const override;

			/**
         	 * @name Functions used in dealing with run-time parameters
         	 * @{
             */
            /**
             * Declare the parameters this class takes through input files.
             */
			static void declare_parameters(ParameterHandler &prm);

			/**
             * Read the parameters this class declares from the parameter file.
             */
			void parse_parameters(ParameterHandler &prm) override;

			void create_additional_named_outputs(MaterialModel::MaterialModelOutputs<dim> &out) const override;

		private:

			/**
			 * List of thermal expansivities.
			 */
			std::vector<double> thermal_expansivities;

			/**
			 * List of reference specific heats.
			 */
        	std::vector<double> reference_specific_heats;

        	/**
        	 * List of thermal conductivities.
        	 */
        	std::vector<double> thermal_conductivities;

        	/**
        	 * List of reference viscosities.
        	 */
        	std::vector<double> eta_0;

        	/**
        	 * List of reference densities.
        	 */
        	std::vector<double> reference_densities;

        	double reference_rho_s;

        	double reference_rho_f;

        	double reference_T;

        	double xi_0;

        	double eta_f;

        	double thermal_viscosity_exponent;

        	double thermal_bulk_viscosity_exponent;

        	double reference_permeability;

        	double alpha_phi;

        	double depletion_density_change;

        	double depletion_solidus_change;

        	double pressure_solidus_change;

        	double surface_solidus;

        	double compressibility;

        	double melt_compressibility;

        	double melting_time_scale;

        	double alpha_depletion;

        	double delta_eta_depletion_max;

        	double peridotite_melting_entropy_change;

        	bool include_melting_and_freezing;

        	MaterialUtilities::CompositionalAveragingOperation viscosity_averaging;

        	virtual double melt_fraction(const double temperature, 
        		                         const double pressure, 
        		                         const double depletion) const;

		};
	}
}

#endif