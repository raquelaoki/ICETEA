import tensorflow as tf


# adapted from https://github.com/Google-Health/genomics-research/blob/bd0d0d0d581d29584a1d203b8f7a44385d0749cb/ml-based-vcdr/learning/model_utils.py#L99


def inceptionv3(model_config, image_shape=(587, 587, 3)):
    """Returns an InceptionV3 architecture as defined by the configuration.
  See https://tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3.
  Args:
    model_config: A dict containing model hyperparamters.
    image_shape
  Returns:
    An InceptionV3-based model.
  """
    input_shape = model_config.get('input_shape', image_shape)

    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=model_config.get('weights', 'imagenet'),
        input_shape=input_shape,
        pooling=model_config.get('pooling', 'avg'))
    weight_decay = model_config.get('weight_decay', 0.0)
    # if weight_decay:
    #  backbone = add_l2_regularizers(
    #      backbone, tf.keras.layers.Conv2D, l2=weight_decay)
    backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

    inputs_image = tf.keras.Input(shape=input_shape, name='image')
    hid = backbone(inputs_image)
    hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)

    outputs = build_outcome_head(model_config, hid, l2=weight_decay)

    model = tf.keras.Model(
        inputs=[inputs_image], outputs=outputs, name='inceptionv3')
    model.summary()
    print(f'Number of l2 regularizers: {len(model.losses)}.')
    return model


def build_outcome_head(model_config, inputs, l2):
    outcome_type = model_config.get('outcome_type', None)
    if outcome_type is None:
        raise ValueError(f'Provided `model_config` missing `outcome_type`: {model_config}')

    l2_regularizer = tf.keras.regularizers.L2(l2) if l2 else None

    if outcome_type == 'regression':
        head = tf.keras.layers.Dense(
            1,
            dtype=tf.float32,
            name='outcome',
            kernel_regularizer=l2_regularizer)
        return head(inputs)

    if outcome_type == 'classification':
        if model_config['num_classes'] < 2:
            raise ValueError('Binary heads should specify `config.num_classes=2`.'
                             'Binary labels are assumed to be one-hot vectors.')

        head = tf.keras.layers.Dense(
            model_config['num_classes'],
            activation='softmax',
            dtype=tf.float32,
            name='outcome',
            kernel_regularizer=l2_regularizer)
        return head(inputs)

    raise ValueError(f'Unknown outcome type: {outcome_type}')


def build_optimizer(opt_config):
    initial_learning_rate = opt_config.get('learning_rate', 0.001)
    steps_per_epoch = 10  # CHANGE HERE LATER

    schedule_config = opt_config.get('schedule', {})
    schedule_type = schedule_config.get('schedule', None)

    if schedule_type == 'exponential':
        decay_steps = int(schedule_config.epochs_per_decay * steps_per_epoch)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=schedule_config.decay_rate,
            staircase=schedule_config.staircase)
    elif schedule_type is None:
        learning_rate = initial_learning_rate
        print(f'No LR schedule provided. Using a fixed LR of "{learning_rate}".')
    else:
        raise ValueError(f'Unknown scheduler name: {schedule_type}')

    opt_type = opt_config.get('optimizer', None)
    if opt_type == 'adam':
        opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=opt_config.get('beta_1', 0.9),
            beta_2=opt_config.get('beta_2', 0.999),
            epsilon=opt_config.get('epsilon', 1e-07),
            amsgrad=opt_config.get('amsgrad', False))
        # start_step = int(steps_per_epoch * 1)
        return opt

    raise ValueError(f'Unknown optimizer name: {opt_type}')


def compile_model(model_config):
    """Builds a graph and compiles a tf.keras.Model based on the configuration."""

    model = inceptionv3(model_config)

    losses = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # add []?
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy')]
    optimizer = build_optimizer(model_config)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics)

    return model



