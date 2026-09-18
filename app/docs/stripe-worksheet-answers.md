# Stripe documentation worksheet answers

These answers address all ten questions in the supplied Stripe Doc Activity worksheet and the current Stripe documentation. The main lesson for NebulaX is to help maintenance staff identify their task, take the next step and understand the result without needing to understand the model implementation.

## Question 1

**C. Software developers building payment systems.** The guide discusses integration, development tools and API authentication. It primarily addresses people adding payments to software, although it also offers a route for users who do not write code. [Stripe getting started](https://docs.stripe.com/get-started)

## Question 2

The page helps someone begin using Stripe: set up an account, choose an integration approach and find the guide for their payment task.

The descriptive title, short introduction, clear section headings and task-based links make its purpose easy to identify in a quick scan. [Stripe getting started](https://docs.stripe.com/get-started)

## Question 3

Use cases connect the documentation to what someone wants to achieve. A reader can recognise a task, such as collecting invoice payments, before knowing which Stripe product or technical feature supports it. This reduces the knowledge needed to choose a starting point. [Stripe getting started](https://docs.stripe.com/get-started)

## Question 4

**C. By user goal.** Guides are grouped around outcomes such as taking payments, offering subscriptions or onboarding accounts. Programming-language choices appear within the relevant interactive guides. [Stripe quickstarts](https://docs.stripe.com/quickstarts)

## Question 5

The four stages are:

1. Configure what you sell, its prices and accepted payment methods.
2. Create the subscription page customers use.
3. Connect the application to Stripe through its API.
4. Test the completed page.

The four numbered stage headings identify the order, with smaller instructions beneath each. [Subscription quickstart](https://docs.stripe.com/billing/quickstart)

## Question 6

**Both.** Explanatory text is static; language selectors, feature switches and the associated code display are interactive.

Static content is easy to scan, print and revisit but cannot adapt to a reader’s choices. Interactive content can show relevant examples and immediate changes, but may hide alternatives and needs accessible controls and a reliable browser. [Subscription quickstart](https://docs.stripe.com/billing/quickstart)

## Question 7

Generally, yes—for its developer audience. Short instructions, action verbs and descriptive headings help readers follow the task. Terms such as API, endpoint and webhook still require technical knowledge, so it is not entirely beginner-friendly for non-developers. [Subscription quickstart](https://docs.stripe.com/billing/quickstart)

## Question 8

Three direct instructions are:

- “Add your products and prices”
- “Add a checkout button”
- “Run the server”

Each begins with a verb and names the action. [Subscription quickstart](https://docs.stripe.com/billing/quickstart)

## Question 9

The table lets readers compare both products against the same criteria without remembering separate paragraphs. For example, it makes the difference between an invoice for a particular customer and a reusable payment link easy to spot. This supports choosing the right tool quickly. [Payment Links comparison](https://docs.stripe.com/payment-links#compare-invoicing-and-payment-links)

## Question 10

The current top navigation tabs switch between documentation areas. Connect belongs to the platforms and marketplaces area; selecting another tab changes the page and its topic navigation. This helps readers focus on relevant guidance and recognise their location. The benefit depends on clear labels because users can overlook content hidden in another area. [Stripe Connect](https://docs.stripe.com/connect)

## Key takeaways for writing user guides

- Begin with the reader’s goal and explain what they will achieve.
- Use action verbs and name the object: tell readers what to select, add or download.
- Present required steps in order and label optional steps clearly.
- Match vocabulary to the audience; explain necessary specialist terms.
- Put file requirements and help beside the action that needs them.
- Use tables for comparisons and interactive controls when choices change the guidance.
- Keep advanced detail available without making it a prerequisite for the main task.
- Explain what results mean, their limits and the next useful action.
- Make errors actionable: say what happened and how to recover.
- Keep labels consistent between navigation, buttons and instructions.

## Changes applied to NebulaX

| Previous wording | Updated wording | Reason |
| --- | --- | --- |
| Control room | Check train condition | Names the user’s task |
| Model registry | Model setup | Makes the purpose easier to recognise |
| Choose a subsystem | Choose what to check | Removes unnecessary technical vocabulary |
| Analyze recording | Check these files | Connects the action to the selected files |
| Rail corrugation | Rail condition | Uses a familiar entry label; explains corrugation in context |
| Explore saved results | View example results | Makes the example route explicit |
| No inference run | Your files were not checked | Explains the distinction in everyday language |
| Suggested next check | What to check next | Gives a direct next step |

The main screen now explains the expected result and what to look for. Model fingerprints and scoring formulas sit inside expandable technical sections. Setup errors identify a recovery step. Official prediction values, model routing and CSV schemas are unchanged.
