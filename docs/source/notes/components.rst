.. _component-modeling:

Component Modeling
==================

The area and energy of components in the architecture:ref:`architecture` either be
specified directly, or by calls to the `HWComponents
<https://github.com/Accelergy-Project/hwcomponents>`_ library.



the ``hwcomponents`` library:

.. _hwcomponents: https://github.com/Accelergy-Project/hwcomponents

Additional information can be added to the component to help with the calculation of the
energy, area, and leakage values. To call the ``hwcomponents`` library, ``component_class``
must be provided, and additional non-underscored:ref:`underscore-discussion` attributes
can be provided to help the calculation.

TODO: REFERENCE A NOTEBOOK

.. code-block:: python

   spec = ff.Spec.from_yaml("arches/tpu_v4i_like.arch.yaml")
   spec.calculate_component_energy_area(
       energy=True,
       area=True,
   )

Total energy of a component is calculated by the sum, across all actions, of the energy
per action times the number of actions incurred as calculated in
:ref:`calculating-num-actions`.