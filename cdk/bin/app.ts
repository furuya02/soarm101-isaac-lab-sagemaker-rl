#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { SoarmStack } from "../lib/stack";

const app = new cdk.App();

const accountId: string =
  app.node.tryGetContext("account_id") ??
  process.env.CDK_DEFAULT_ACCOUNT ??
  (() => {
    throw new Error("account_id context or CDK_DEFAULT_ACCOUNT env var is required");
  })();

const region: string =
  app.node.tryGetContext("region") ??
  process.env.CDK_DEFAULT_REGION ??
  "ap-northeast-1";

const bucketSuffix: string =
  app.node.tryGetContext("bucket_suffix") ?? accountId;

const enableBudget: boolean =
  (app.node.tryGetContext("enable_budget") ?? "true").toString() === "true";

const budgetEmail: string =
  app.node.tryGetContext("budget_email") ?? "please-set-budget-email@example.com";

new SoarmStack(app, "SoarmStack", {
  env: { account: accountId, region },
  projectName: "soarm101-isaac-lab-sagemaker-rl",
  bucketSuffix,
  enableBudget,
  budgetEmail,
});
