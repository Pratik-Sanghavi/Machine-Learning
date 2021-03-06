?	K? ?r@K? ?r@!K? ?r@	(??G}???(??G}???!(??G}???"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0K? ?r@?????@1ی?ոq@I???1v)@Y????Д??r0*	5^?I8?@2X
!Iterator::Root::Prefetch::BatchV2?fF????!P???ďU@)??????1?%?Y<>@:Preprocessing2a
*Iterator::Root::Prefetch::BatchV2::Shuffle@?%:?,???!Y?,2]L@)?DeÚ???1??	ɍ8@:Preprocessing2z
CIterator::Root::Prefetch::BatchV2::Shuffle::FiniteSkip::TensorSlice@??/g???!?#???x0@)??/g???1?#???x0@:Preprocessing2m
6Iterator::Root::Prefetch::BatchV2::Shuffle::FiniteSkip@???4??!?O?,??@)jO?9????1?fg>?.@:Preprocessing2E
Iterator::Root????u??!??؁+@)????w??1?͖"^D @:Preprocessing2O
Iterator::Root::Prefetch??J????!ݯ???z@)??J????1ݯ???z@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9)??G}???I????n`@Q1?[?gW@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ??!       "	ی?ոq@ی?ոq@!ی?ոq@*      ??!       2      ??!       :	???1v)@???1v)@!???1v)@B      ??!       J	????Д??????Д??!????Д??R      ??!       Z	????Д??????Д??!????Д??b      ??!       JGPUY)??G}???b q????n`@y1?[?gW@?"d
8gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter;?f?0??!;?f?0??0"d
8gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????lO??!??-Q'@??0"5
model/conv2d_6/Relu_FusedConv2D??cE???!?r?%A??"b
7gradient_tape/model/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput??O?=???!?Xjy4 ??0"5
model/conv2d_5/Relu_FusedConv2D?y"VѴ?!D??4??"b
7gradient_tape/model/conv2d_5/Conv2D/Conv2DBackpropInputConv2DBackpropInputw0?a??!???&??0"d
8gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??_?e???!s?M?%??0"5
model/conv2d_2/Relu_FusedConv2DN??)?V??!?V?????"d
8gradient_tape/model/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterW??cW???!ݕ'8%??0"b
7gradient_tape/model/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInputV?c>??!b'g &??0I?r<?ya?Q?????X@Y????X8@a??۩?R@q???i?E@yӔli3K\?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?43.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Kepler)(: B 