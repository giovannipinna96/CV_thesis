def loss_fn_training_op(self, x, y, z, logits, x_recon, class_means):
    """ Computes the loss functions and creates the update ops.

    :param x - input X
    :param y - labels y
    :param z - z layer transform of X.
    :param logits - softmax logits if ce loss is used. Can be None if only ii-loss.
    :param recon - reconstructed X. Experimental! Can be None.
    :class_means - the class means.
    """
    # Calculate intra class and inter class distance
    if self.dist == 'class_mean':  # For experimental pupose only
        self.intra_c_loss, self.inter_c_loss = self.inter_intra_diff(
            z, tf.compat.v1.cast(y, tf.int32), class_means)
    elif self.dist == 'all_pair':  # For experimental pupose only
        self.intra_c_loss, self.inter_c_loss = self.all_pair_inter_intra_diff(
            z, tf.cast(y, tf.int32))
    elif self.dist == 'mean_separation_spread':  # ii-loss
        self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spred(
            z, tf.cast(y, tf.int32), class_means)
    elif self.dist == 'min_max':  # For experimental pupose only
        self.intra_c_loss, self.inter_c_loss = self.inter_min_intra_max(
            z, tf.cast(y, tf.int32), class_means)

    # Calculate reconstruction loss
    if self.enable_recon_loss:  # For experimental pupose only
        self.recon_loss = tf.reduce_mean(tf.squared_difference(x, x_recon))

    if self.enable_intra_loss and self.enable_inter_loss:  # The correct ii-loss
        self.loss = tf.compat.v1.reduce_mean(self.intra_c_loss - self.inter_c_loss)
    elif self.enable_intra_loss and not self.enable_inter_loss:  # For experimental pupose only
        self.loss = tf.compat.v1.reduce_mean(self.intra_c_loss)
    elif not self.enable_intra_loss and self.enable_inter_loss:  # For experimental pupose only
        self.loss = tf.compat.v1.reduce_mean(-self.inter_c_loss)
    elif self.div_loss:  # For experimental pupose only
        self.loss = tf.compat.v1.reduce_mean(self.intra_c_loss / self.inter_c_loss)
    else:  # For experimental pupose only
        self.loss = tf.reduce_mean((self.recon_loss * 1. if self.enable_recon_loss else 0.)
                                   + (self.intra_c_loss * 1. if self.enable_intra_loss else 0.)
                                   - (self.inter_c_loss * 1. if self.enable_inter_loss else 0.)
                                   )

    # Classifier loss
    if self.enable_ce_loss:
        self.ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

    tvars = tf.compat.v1.trainable_variables()
    e_vars = [var for var in tvars if 'enc_' in var.name]
    classifier_vars = [var for var in tvars if 'enc_' in var.name or 'classifier_' in var.name]
    recon_vars = [var for var in tvars if 'enc_' in var.name or 'dec_' in var.name]

    # Training Ops
    # TODO necessita di queste variabili già inizializzate in fase di training (fit)
    # if self.is_training:
    # with tf.GradientTape(persistent=True) as tape:
    if self.enable_recon_loss:
        self.recon_train_op = self.recon_opt.minimize(self.recon_loss, var_list=recon_vars)

    if self.enable_inter_loss or self.enable_intra_loss or self.div_loss:
        # self.train_op = tape.gradient(self.loss, e_vars)
        # self.train_op = self.opt.minimize(self.loss, var_list=e_vars, tape=tf.GradientTape(persistent=False))
        self.train_op = self.opt.minimize(self.loss, var_list=e_vars)

    if self.enable_ce_loss:
        # self.ce_train_op = tape.gradient(self.loss, classifier_vars)
        # self.ce_train_op = self.c_opt.minimize(self.ce_loss, var_list=classifier_vars,
        #                                       tape=tf.GradientTape(persistent=False))
        self.ce_train_op = self.c_opt.minimize(self.ce_loss, var_list=classifier_vars)



    @tf.function
    def inter_intra_diff(self, data, labels, class_mean):
        """ Calculates the intra-class and inter-class distance
        as the average distance from the class means.
        """
        sq_diff = self.sq_difference_from_mean(data, class_mean)

        inter_intra_sq_diff = self.bucket_mean(sq_diff, labels, 2)
        inter_class_sq_diff = inter_intra_sq_diff[0]
        intra_class_sq_diff = inter_intra_sq_diff[1]
        return intra_class_sq_diff, inter_class_sq_diff


    def sq_difference_from_mean(self, data, class_mean):
        """ Calculates the squared difference from clas mean.
        """
        sq_diff_list = []
        for i in range(self.y_dim):
            sq_diff_list.append(tf.reduce_mean(
                tf.compat.v1.squared_difference(data, class_mean[i]), axis=1))

        return tf.stack(sq_diff_list, axis=1)


    def bucket_mean(self, data, bucket_ids, num_buckets):
        total = tf.compat.v1.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.compat.v1.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count