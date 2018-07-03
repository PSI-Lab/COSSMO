import tensorflow as tf
from tensorflow.contrib.training import bucket_by_sequence_length
from itertools import repeat


def convert_to_tf_example(const_exonic_seq, const_intronic_seq,
                          alt_exonic_seq, alt_intronic_seq,
                          psi_distribution, psi_std,
                          alt_ss_position, alt_ss_type,
                          const_site_id, const_site_position,
                          n_alt_ss, event_type,
                          strand=None, coord_sys='dna0'):
    """Encode a COSSMO example as a TFRecord. 
    
    coord_sys must be either 'rna1' or 'dna0'. if 'dna0' then strand
    must be either '+' or '-'. 
    """

    assert len(alt_exonic_seq) == n_alt_ss
    assert len(alt_intronic_seq) == n_alt_ss
    # assert len(psi_distribution) == n_alt_ss
    # assert len(psi_std) == n_alt_ss
    assert event_type in ['acceptor', 'donor']
    assert len(alt_ss_type) == n_alt_ss
    assert all([t in ('annotated', 'gtex', 'maxent', 'hard_negative', 'dinuc')
                for t in alt_ss_type])
    assert coord_sys in ['dna0', 'rna1']

    feature = {
            'n_alt_ss': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[n_alt_ss])
            ),
            'event_type': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[event_type])
            ),
            'const_seq': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[const_exonic_seq, const_intronic_seq])
            ),
            'const_site_id': tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[const_site_id])
            ),
            'const_site_position': tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[const_site_position])
            ),
        }

    # strand only included for dna0, else we maintain compat with previous rna1 data set format
    if coord_sys == 'dna0':
        assert strand in ['+', '-']
        feature['strand'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[strand])
            )

    feature_list = {
            'alt_seq': tf.train.FeatureList(
                feature=[tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[aes, ais])
                ) for aes, ais in
                         zip(alt_exonic_seq, alt_intronic_seq)]
            ),
            'psi': tf.train.FeatureList(
                feature=[tf.train.Feature(
                    float_list=tf.train.FloatList(value=psi))
                         for psi in psi_distribution]),
            'psi_std': tf.train.FeatureList(
                feature=[tf.train.Feature(
                    float_list=tf.train.FloatList(value=psi_sd))
                         for psi_sd in psi_std]),
            'alt_ss_position': tf.train.FeatureList(
                feature=[tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[pos]))
                         for pos in alt_ss_position]),
            'alt_ss_type': tf.train.FeatureList(
                feature=[tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[t]))
                        for t in alt_ss_type])
        }

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=feature),
        feature_lists=tf.train.FeatureLists(feature_list=feature_list)
    )

    return example


def read_single_cossmo_example(serialized_example, n_tissues, coord_sys='rna1'):
    """Decode a single COSSMO example
    
    coord_sys must be one of 'rna1' or 'dna0', if 'dna0' then an extra 'strand' field
    must exist in the tfrecord and is extracted.
    """

    assert coord_sys in ['dna0', 'rna1']

    context_features = {
        'n_alt_ss': tf.FixedLenFeature([], tf.int64),
        'event_type': tf.FixedLenFeature([], tf.string),
        'const_seq': tf.FixedLenFeature([2], tf.string),
        'const_site_id': tf.FixedLenFeature([], tf.string),
        'const_site_position': tf.FixedLenFeature([], tf.int64),
    }

    if coord_sys == 'dna0':
        context_features['strand'] = tf.FixedLenFeature([], tf.string)

    sequence_features = {
        'alt_seq': tf.FixedLenSequenceFeature([2], tf.string),
        'psi': tf.FixedLenSequenceFeature([n_tissues], tf.float32),
        'psi_std': tf.FixedLenSequenceFeature([n_tissues], tf.float32),
        'alt_ss_position': tf.FixedLenSequenceFeature([], tf.int64),
        'alt_ss_type': tf.FixedLenSequenceFeature([], tf.string)
    }

    decoded_features = tf.parse_single_sequence_example(
        serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return decoded_features


def dynamic_bucket_data_pipeline(input_files, num_epochs, buckets, batch_sizes,
                                 alt_ss_type, n_tissues, shuffle=True,
                                 sort=True, max_n_alt_ss=2000):

    """
    Implements the input data pipeline. Given a set of files containing
    TFRecords with COSSMO examples, this function returns a queue that
    produces mini-batches of the desired size, ready to be input into COSSMO.

    Each minibatch contains examples from only one bucket. The buckets are
    defined in terms of the number of alternative splice sites.

    Args:
        input_files: A list of input files
        num_epochs: The number of epochs
        buckets: An array defining a set of intervals for the buckets. The
            number of buckets is ``len(buckets) - 1``.
        batch_sizes: An array containing the batch size for each bucket. Must
            be of length `len(buckets) - 1` or an integer in which the
            mini-batches will all be of the same size.
        alt_ss_type: ``"acceptor"` or `"donor"``

    Returns:
        A dictionary containing the following tensors:
         - n_alt_ss
         - alt_ss_type
         - psi
         - const_seq
         - const_dna_seq
         - alt_seq
         - alt_dna_seq
         - rna_seq
    """

    if buckets[0] == 0:
        buckets = buckets[1:]

    decoded_example = read_data_files(
        alt_ss_type, input_files, n_tissues, num_epochs, shuffle, sort)

    with tf.name_scope('data_pipeline'):
        n_alt_ss = decoded_example['n_alt_ss']

        combined_queue = bucket_by_sequence_length(
            n_alt_ss,
            decoded_example,
            batch_sizes,
            buckets,
            dynamic_pad=True,
            allow_smaller_final_batch=True,
            keep_input=n_alt_ss <= max_n_alt_ss
        )[1]

        combined_queue['psi'] = \
            tf.transpose(combined_queue['psi'], [2, 0, 1])
        combined_queue['psi_std'] = \
            tf.transpose(combined_queue['psi_std'], [2, 0, 1])

    return combined_queue


def read_data_files(alt_ss_type, input_files, n_tissues,
                    num_epochs=None, shuffle=False, sort=True):

    with tf.name_scope('data_pipeline'):
        assert (alt_ss_type in ('acceptor', 'donor'))
        filename_queue = tf.train.string_input_producer(
            input_files, num_epochs=num_epochs, shuffle=shuffle)
        file_reader = tf.TFRecordReader()
        tf_record_key, serialized_example = file_reader.read(filename_queue)
        _decoded_example = read_single_cossmo_example(serialized_example,
                                                      n_tissues)

        decoded_example = _decoded_example[0]
        decoded_example.update(_decoded_example[1])

        if sort:
            sorted_distance_indices = tf.nn.top_k(-tf.abs(decoded_example['alt_ss_position'] -
                                                          decoded_example['const_site_position']),
                                                  k=tf.cast(decoded_example['n_alt_ss'], tf.int32),
                                                  sorted=True).indices
            decoded_example['alt_seq'] = tf.gather(decoded_example['alt_seq'], sorted_distance_indices)
            decoded_example['psi'] = tf.gather(decoded_example['psi'], sorted_distance_indices)
            decoded_example['psi_std'] = tf.gather(decoded_example['psi_std'], sorted_distance_indices)
            decoded_example['alt_ss_position'] = tf.gather(decoded_example['alt_ss_position'], sorted_distance_indices)
            decoded_example['alt_ss_type'] = tf.gather(decoded_example['alt_ss_type'], sorted_distance_indices)

        decoded_example['tfrecord_key'] = tf_record_key
        const_exonic_seq, const_intronic_seq = \
            tf.split(axis=0, num_or_size_splits=2, value=decoded_example['const_seq'])
        alt_exonic_seq, alt_intronic_seq = \
            tf.split(axis=1, num_or_size_splits=2, value=decoded_example['alt_seq'])
        alt_exonic_seq = tf.squeeze(alt_exonic_seq, [1])
        alt_intronic_seq = tf.squeeze(alt_intronic_seq, [1])
        const_exonic_seq = tf.decode_raw(const_exonic_seq, tf.uint8)
        const_intronic_seq = tf.decode_raw(const_intronic_seq, tf.uint8)
        alt_exonic_seq = tf.decode_raw(alt_exonic_seq, tf.uint8)
        alt_intronic_seq = tf.decode_raw(alt_intronic_seq, tf.uint8)
        tile_multiples = tf.stack(
            [tf.to_int32(decoded_example['n_alt_ss']), 1])
        const_exonic_seq_tiled = tf.tile(
            const_exonic_seq, tile_multiples
        )
        if alt_ss_type == 'acceptor':
            rna_seq = tf.concat(axis=1, values=[const_exonic_seq_tiled, alt_exonic_seq])
            const_dna = tf.squeeze(
                tf.concat(axis=1, values=[const_exonic_seq, const_intronic_seq]),
                [0])
            alt_dna = tf.concat(axis=1, values=[alt_intronic_seq, alt_exonic_seq])
        elif alt_ss_type == 'donor':
            rna_seq = tf.concat(axis=1, values=[alt_exonic_seq, const_exonic_seq_tiled])
            const_dna = tf.squeeze(
                tf.concat(axis=1, values=[const_intronic_seq, const_exonic_seq]),
                [0])
            alt_dna = tf.concat(axis=1, values=[alt_exonic_seq, alt_intronic_seq])
        decoded_example['rna_seq'] = rna_seq
        decoded_example['const_dna_seq'] = const_dna
        decoded_example['alt_dna_seq'] = alt_dna
        decoded_example['n_alt_ss'] = tf.to_int32(
            decoded_example['n_alt_ss'], 'n_alt_ss')
    return decoded_example


def read_from_placeholders(alt_ss_type):
    assert alt_ss_type in ('acceptor', 'donor')

    with tf.name_scope('data_pipeline'):
        placeholders = {
            'rna_seq': tf.placeholder(tf.string, [None, None], name='rna_seq'),
            'alt_dna_seq': tf.placeholder(tf.string, [None, None], name='alt_dna_seq'),
            'const_dna_seq': tf.placeholder(tf.string, [None], name='const_dna_seq'),
            'const_site_position': tf.placeholder(tf.int32, [None], name='const_site_position'),
            'alt_ss_position': tf.placeholder(tf.int32, [None, None], name='alt_ss_position'),
            'n_alt_ss': tf.placeholder(tf.int32, [None], name='n_alt_ss'),
        }

        cossmo_inputs = {
            'rna_seq': tf.decode_raw(placeholders['rna_seq'], tf.uint8),
            'alt_dna_seq':
                tf.decode_raw(placeholders['alt_dna_seq'], tf.uint8),
            'const_dna_seq':
                tf.decode_raw(placeholders['const_dna_seq'], tf.uint8),
            'const_site_position': placeholders['const_site_position'],
            'alt_ss_position': placeholders['alt_ss_position'],
            'n_alt_ss': placeholders['n_alt_ss']
        }

        return placeholders, cossmo_inputs


def sequential_data_pipeline(input_files, num_epochs, batch_size,
                             alt_ss_type, n_tissues, shuffle=False, sort=True,
                             **kwargs):
    example = read_data_files(alt_ss_type, input_files, n_tissues,
                              num_epochs, shuffle, sort)

    if batch_size > 1:
        batches = tf.train.batch(example, batch_size, enqueue_many=False,
                                 dynamic_pad=True, **kwargs)
    else:
        batches = example
        for k, v in batches.iteritems():
            batches[k] = tf.expand_dims(v, 0)

    batches['psi'] = \
        tf.transpose(batches['psi'], [2, 0, 1])
    batches['psi_std'] = \
        tf.transpose(batches['psi_std'], [2, 0, 1])

    return batches


def make_pipeline(configuration, data_pipeline, input_files):
    """Sets up the input data pipeline"""

    if data_pipeline == 'dynamic':
        batches = dynamic_bucket_data_pipeline(
            input_files,
            configuration['num_epochs'],
            configuration['buckets'],
            configuration['batch_sizes'],
            configuration['event_type'], 1, sort=configuration.get('sort_ss_by_position', True)

        )
    elif data_pipeline == 'sequential':
        batches = sequential_data_pipeline(
            input_files,
            configuration['num_epochs'],
            configuration['batch_size'],
            configuration['event_type'], 1, sort=configuration.get('sort_ss_by_position', True)
        )

    del batches['alt_seq']
    # del batches['alt_ss_type']
    del batches['const_seq']
    del batches['const_site_id']
    del batches['event_type']
    del batches['psi_std']
    return batches
