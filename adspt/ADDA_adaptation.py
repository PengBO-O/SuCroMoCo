from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import get_scheduler

from itertools import zip_longest


def adapting(src_encoder, tgt_encoder, discriminator,
             src_loader, tgt_loader, accelerator, args, logger):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    # tgt_encoder.train()
    # discriminator.train()

    steps = min(len(tgt_loader), len(src_loader)) * args.adapt_epoch

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer_tgt = optim.AdamW(filter(lambda p: p.requires_grad, tgt_encoder.parameters()), lr=args.gen_lr)
    schedular_tgt = get_scheduler("cosine",
                                  optimizer=optimizer_tgt,
                                  num_warmup_steps=steps * 0.1,
                                  num_training_steps=steps
                                  )

    optimizer_dis = optim.AdamW(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.dis_lr)
    schedular_dis = get_scheduler("cosine",
                                  optimizer=optimizer_dis,
                                  num_warmup_steps=steps * 0.1,
                                  num_training_steps=steps
                                  )

    optimizer_tgt, schedular_tgt, \
        optimizer_dis, schedular_dis = accelerator.prepare(optimizer_tgt, schedular_tgt,
                                                           optimizer_dis, schedular_dis)

    ####################
    # 2. train network #
    ####################
    progress_bar = tqdm(range(steps))

    for param in src_encoder.parameters():
        param.requires_grad = False

    for epoch in range(args.adapt_epoch):
        src_encoder.eval()
        tgt_encoder.train()
        discriminator.train()
        for step, (src_batch, tgt_batch) in enumerate(zip(src_loader, tgt_loader)):
            ###########################
            # 2.1 train discriminator #
            ###########################
            optimizer_dis.zero_grad()

            src_inputs = {k: src_batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
            src_inputs['use_prompt'] = True

            tgt_inputs = {k: tgt_batch[k] for k in ['input_ids', 'attention_mask', 'sep_locations']}
            tgt_inputs['use_prompt'] = True

            # extract and concat features
            outputs_src = src_encoder(**src_inputs)
            feat_src = outputs_src['hidden']
            pred_src = discriminator(feat_src)

            outputs_tgt = tgt_encoder(**tgt_inputs)
            feat_tgt = outputs_tgt['hidden']
            pred_tgt = discriminator(feat_tgt.detach())

            pred = torch.cat((pred_src, pred_tgt), 0)

            label_src = torch.ones(feat_src.size(0)).long().cuda()
            label_tgt = torch.zeros(feat_tgt.size(0)).long().cuda()
            label = torch.cat((label_src, label_tgt), 0)
            loss_dis = criterion(pred, label)

            loss_dis.backward()
            optimizer_dis.step()
            schedular_dis.step()

            pred_cls = torch.squeeze(pred.max(1)[1])
            acc = (pred_cls == label).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################
            # zero gradients for optimizer
            optimizer_tgt.zero_grad()

            # extract and target features

            outputs_tgt = tgt_encoder(**tgt_inputs)
            feat_tgt = outputs_tgt['hidden']

            # predict on discriminator
            pred_tgt = discriminator(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().cuda()

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()
            optimizer_tgt.step()
            schedular_tgt.step()

            gen_pred = torch.squeeze(pred_tgt.max(1)[1])
            gen_acc = (gen_pred == label_tgt).float().mean()
            if (step + 1) % args.print_step == 0:
                tgt_lr = optimizer_tgt.state_dict()['param_groups'][0]['lr']

                logger.info(f"generator learning rate: {tgt_lr:.3e}")
                logger.info(f"generator loss: {loss_tgt.item():.5f}")
                logger.info(f"generator acc: {gen_acc.item() * 100:.2f}")
                logger.info('-' * 20)

                dis_lr = optimizer_dis.state_dict()['param_groups'][0]['lr']

                logger.info(f"discriminator learning rate: {dis_lr:.3e}")
                logger.info(f'discriminator loss: {loss_dis.item():.5f}')
                logger.info(f"discriminator acc: {acc.item() * 100:.2f}")
                logger.info('-' * 20)

            progress_bar.update(1)
        logger.info('=' * 30)
        logger.info(f"End of Epoch {epoch}")

    return tgt_encoder

    #############################
    # 2.4 save model argseters #
    #############################
    # if (epoch + 1) % args.save_step == 0:
    #     torch.save(critic.state_dict(), os.path.join(
    #         args.model_root,
    #         "adspt-critic-{}.pt".format(epoch + 1)))
    #     torch.save(tgt_encoder.state_dict(), os.path.join(
    #         args.model_root,
    #         "adspt-target-encoder-{}.pt".format(epoch + 1)))

    # torch.save(critic.state_dict(), os.path.join(
    #     args.model_root,
    #     "adspt-critic-final.pt"))
    # torch.save(tgt_encoder.state_dict(), os.path.join(
    #     args.model_root,
    #     "adspt-target-encoder-final.pt"))
