	,-#??*?@,-#??*?@!,-#??*?@	19??c32?19??c32?!19??c32?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$,-#??*?@j???vD@A?C5%?)?@Y???Qٰ??*	?Z??2xA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator4?????@!?????X@)4?????@1?????X@:Preprocessing2F
Iterator::Model3?ۃ??!e???WG(?)?[?J???1??&?(??:Preprocessing2P
Iterator::Model::Prefetch?
?lw??!cܿ???)?
?lw??1cܿ???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?????@!P???X@)??Pk?wl?1y*7v??>:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no919??c32?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	j???vD@j???vD@!j???vD@      ??!       "      ??!       *      ??!       2	?C5%?)?@?C5%?)?@!?C5%?)?@:      ??!       B      ??!       J	???Qٰ?????Qٰ??!???Qٰ??R      ??!       Z	???Qٰ?????Qٰ??!???Qٰ??JCPU_ONLYY19??c32?b 