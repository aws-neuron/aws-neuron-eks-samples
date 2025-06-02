#!/bin/bash
rm -rf cdk.* package* node_modules/
npm install -g aws-cdk
npm install aws-cdk-lib
npm install ts-node typescript 
npm install typescript --save-dev
npx tsc --init
npx tsc
. ~/.bash_profile
cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION
npm install
cdk deploy --app "npx ts-node --prefer-ts-exts ./pipeline.ts"  --parameters BASEIMAGEAMDXLATAG=$BASE_IMAGE_AMD_XLA_TAG --parameters BASEIMAGEAMDCUDTAG=$BASE_IMAGE_AMD_CUD_TAG  --parameters BASEREPO=$BASE_REPO --parameters IMAGEAMDXLATAG=$IMAGE_AMD_XLA_TAG --parameters IMAGEAMDCUDTAG=$IMAGE_AMD_CUD_TAG --parameters GITHUBREPO=$GITHUB_REPO --parameters GITHUBUSER=$GITHUB_USER --parameters GITHUBBRANCH=$GITHUB_BRANCH --parameters GITHUBOAUTHTOKEN=$GITHUB_OAUTH_TOKEN --parameters BASEIMAGEARMCPUTAG=$BASE_IMAGE_ARM_CPU_TAG --parameters IMAGEARMCPUTAG=$IMAGE_ARM_CPU_TAG
