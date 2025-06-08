import tensorflow as tf
from utilss.classes.adversarial_attacks.adversarial_attack import AdversarialAttack

class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent attack implementation.
    PGD is one of the strongest first-order attacks, iteratively taking gradient
    steps and projecting back onto an ε-ball around the original image.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    def __init__(self, **kwargs):
        super().__init__(
            epsilon=kwargs.get("epsilon") if kwargs.get("epsilon") is not None else 0.7,
            alpha=kwargs.get("alpha") if kwargs.get("alpha") is not None else 0.1,
            num_steps=kwargs.get("num_steps") if kwargs.get("num_steps") is not None else 70
            )

    @tf.function
    def _attack_step(self, model, x_adv, x_orig, target):
        for _ in tf.range(self.num_steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                pred = model(x_adv)
                loss = tf.keras.losses.categorical_crossentropy(target, pred)
            grad = tape.gradient(loss, x_adv)
            x_adv = x_adv - self.alpha * tf.sign(grad)
            x_adv = tf.clip_by_value(x_adv, x_orig - self.epsilon, x_orig + self.epsilon)
            x_adv = tf.clip_by_value(x_adv, -123.68, 151.061)
        return x_adv

    def attack(self, model, image):
        # Get number of classes
        num_classes = model.output_shape[1]
        current_pred = model(image)
        original_class_idx = tf.argmax(current_pred[0]).numpy()
        target_class_idx = (original_class_idx + num_classes // 2) % num_classes
        target = tf.one_hot(target_class_idx, num_classes)
        target = tf.expand_dims(target, axis=0)
        x_orig = tf.identity(image)
        x_adv = tf.identity(image)
        x_adv = self._attack_step(model, x_adv, x_orig, target)
        x_adv = tf.squeeze(x_adv, axis=0)
        return x_adv