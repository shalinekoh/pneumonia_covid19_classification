	ū?m??@ū?m??@!ū?m??@	???|?v4????|?v4?!???|?v4?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ū?m??@?Ӻ
@A?#+?+?@Y??6?Nx??*	????zxA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??!V?@!?N????X@)??!V?@1?N????X@:Preprocessing2F
Iterator::Model??? v??!?%?
a.?)???3K??1.?xq?#?:Preprocessing2P
Iterator::Model::Prefetch+0du????!m!?2??)+0du????1m!?2??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapE(bV?@!???3??X@)???c> p?1:?x??>:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???|?v4?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Ӻ
@?Ӻ
@!?Ӻ
@      ??!       "      ??!       *      ??!       2	?#+?+?@?#+?+?@!?#+?+?@:      ??!       B      ??!       J	??6?Nx????6?Nx??!??6?Nx??R      ??!       Z	??6?Nx????6?Nx??!??6?Nx??JCPU_ONLYY???|?v4?b 