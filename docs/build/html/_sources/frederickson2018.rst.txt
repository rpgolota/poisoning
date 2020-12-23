frederickson2018 API
====================

The :class:`~poisoning.frederickson2018` class is a subclass of :class:`~poisoning.xiao2018`, meaning it has all the data that xiao2018 has.
The reason to use frederickson2018 is to create smarter attack points that are less detectable by staying closer to the original data. Where xiao2018 uses just the bounding box and some arbitrary constants to bound its final attack points, frederickson2018 uses a loss term in order to accomplish a better boundary.

To get started using it, import it from poisoning::

    from poisoning import frederickson2018

Parameters
----------

The following are parameters to the class.

* :obj:`~poisoning.frederickson2018.phi` is a float used as the penalty term.
* :obj:`~poisoning.frederickson2018.power` is a float used to raise the distance of the k-th nearest neighbor to some power.
* :obj:`~poisoning.frederickson2018.k` is an integer, representing the k-th nearest point that is an element in the surrogate dataset.
* :obj:`~poisoning.frederickson2018.outlier_type` is a string representing the type of outlier type that will be used in the algorithm..

Attributes
----------

* :obj:`~poisoning.frederickson2018.outlier_type` string representing the outlier type.

Methods
-------

There are two methods available to the user.

* :meth:`~poisoning.frederickson2018.run`
* :meth:`~poisoning.frederickson2018.autorun`

The difference between the two methods is that autorun does not need to take as an input initial attack points and initial labels for those attack points.

Instead, it takes in a number (int or float[0.0, 1.0]) representing either the number of attacks to create, or the percent of attacks to create.