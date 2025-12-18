API Guide
=========

The API has the following parts:

- The ``frontend`` package contains all parts of the API that are used to parse and
  modify input specifications. The input specification is handled by a top-level
  :py:class:`~fastfusion.frontend.specification.Specification` class. Each attribute of
  the specification is another class that represents a different part of the input, and
  having its own module in the ``frontend`` package.
- The ``mapper`` package contains all parts of the API that are used to map workloads
  onto architectures. If you have a workload and architecture and would like to evaluate
  energy, latency, or other metrics, you can use the ``mapper`` package to do so.
