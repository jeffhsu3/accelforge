.. _component-modeling:

Modeling Component Energy and Area
==================================

The energy and area of components in the architecture:ref:`architecture` can either be
specified directly, or by calls to the `HWComponents
<https://github.com/Accelergy-Project/hwcomponents>`_ library.

Calculating Energy and Area
---------------------------

Component energy and area calculations will populate the following fields for each
component:

- ``attributes.area``: :docstring:`fastfusion.frontend.arch.Component.attributes.area`
- ``attributes.leak_power``: :docstring:`fastfusion.frontend.arch.Component.attributes.leak_power`
- ``actions[<action name>].arguments.energy``: :docstring:`fastfusion.frontend.arch.Action.arguments.energy`
- ``attributes.total_area``: :docstring:`fastfusion.frontend.arch.Component.attributes.total_area`
- ``attributes.total_leak_power``: :docstring:`fastfusion.frontend.arch.Component.attributes.total_leak_power`
- ``attributes.energy_area_log``: :docstring:`fastfusion.frontend.arch.Component.attributes.energy_area_log`
- ``component_model``: :docstring:`fastfusion.frontend.arch.Component.component_model`

The energy and area of a all components in the architecture can be calculated by calling
:py:meth:`~fastfusion.spec.Spec.calculate_component_energy_area`.

.. include-notebook:: notebooks/tutorials/component_energy_area.ipynb
   :name: spec_energy_area
   :language: python

We can also calculate the energy and area of individual components by calling
:py:meth:`~fastfusion.arch.Component.calculate_energy_area` on them.

.. include-notebook:: notebooks/tutorials/component_energy_area.ipynb
   :name: single_component_energy_area
   :language: python

There are additional `Spec.config` fields that affect the energy and area
calculations:

.. include-attrs:: fastfusion.frontend.config.Config

Specifying Energy and Area
---------------------------

One way to specify the area and energy of each component is to directly set the
``attributes.area``, ``attributes.leak_power``, or ``actions[<action
name>].arguments.energy`` fields. The following example from the TPU v4i example
architecture shows uses this approach:

.. include-yaml:: examples/arches/tpu_v4i_like.arch.yaml
   :startfrom: GlobalBuffer
   :same-indent:

If any value is omitted, it will raise an appropriate error when
:py:obj:`~fastfusion.spec.Spec.calculate_component_energy_area` is called, so you may call this
function to check whether you've missed anything. ``hwcomponents`` is invoked
automatically if any of the fields are missing. If you don't want it to be called, then
you can do one of the following:

- If calling :py:obj:`~fastfusion.spec.Spec.calculate_component_energy_area`, then you
  can set ``spec.config.component_models`` and
  ``spec.config.use_installed_component_models`` to an empty list and ``False``,
  respectively.
- If calling :py:obj:`~fastfusion.arch.Component.calculate_energy_area`, then you can set ``models`` to an
  empty list.


Using the ``hwcomponents`` Library
-----------------------------------

``hwcomponents`` is invoked automatically when area and energy are not specified. The
following shows the fields used by ``hwcomponents``:

.. include:: ../../../examples/misc/component_annotated.yaml
   :code: yaml

When ``hwcomponents`` has been used to calculate the energy and area of a component,
then the ``component_model`` field will be set to the `hwcomponents` model used to
calculate the energy and area.

In addition to looking at the ``energy_area_log`` field, we can further inspect the
``component_model`` field to see more information about the model.

.. include-notebook:: notebooks/tutorials/component_energy_area.ipynb
   :name: hwcomponents
   :language: python
