import torch


class DANNCE:

    @classmethod
    def train_adversarial_examples(
        cls,
        input_features: torch.Tensor,
        d_labels: torch.Tensor,
        adversarial_examples_ratio: float,
        adversarial_examples_lr: float,
        adversarial_train_steps: int,
        adv_kl_weight: float,
        model: torch.nn.Module,
        domain_adversary: torch.nn.Module,
        domain_adversary_weight: float,
        logger = None
    ) -> torch.Tensor:
        
        # randomly select the features for update
        adv_mask = torch.bernoulli(adversarial_examples_ratio * torch.ones(input_features.size(dim=0))).type_as(input_features).bool()
        adv_input_features = input_features[adv_mask]
        adv_d_labels = d_labels[adv_mask]
        adv_batch_size = adv_input_features.size(dim=0)

        if adv_input_features.nelement() > 0:

            # get old class distribution
            old_class_distr = torch.nn.functional.log_softmax(model(adv_input_features).logits.detach(), dim=-1)

            # promote the inputs to parameters for optimization
            adv_input_features = torch.nn.Parameter(adv_input_features)
            adv_input_features.requires_grad = True
            optim = torch.optim.Adam(
                [adv_input_features],
                lr = adversarial_examples_lr,
                weight_decay = 0.0
            )

            for i_ in range(adversarial_train_steps):

                # reset gradients
                optim.zero_grad()

                # forward through the model to get features and new class distribution
                kws_output = model(adv_input_features)
                new_class_distr = torch.nn.functional.log_softmax(kws_output.logits, dim=-1)

                # get discriminator loss
                d_loss = domain_adversary(
                    input_features = kws_output.features, 
                    labels = adv_d_labels, 
                    use_grad_reverse = False
                ).loss * domain_adversary_weight
                
                # calculate KL divergence loss for regularization
                kl_loss = adv_kl_weight * torch.nn.functional.kl_div(old_class_distr, new_class_distr, log_target=True)

                # and backpropagate the final loss
                (d_loss + kl_loss).backward()

                # log losses for monitoring
                if logger is not None:
                    logger.log('dannce/domain_loss', d_loss, batch_size=adv_batch_size, sync_dist=True)
                    logger.log('dannce/kl_loss', kl_loss, batch_size=adv_batch_size, sync_dist=True)
                
                # optimizer step
                optim.step()        

        input_features[adv_mask] = adv_input_features.data.detach()

        return input_features