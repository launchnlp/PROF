import torch
from llamafactory.data.utils import Role
from llamafactory.data.template import templates, get_template_and_fix_tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Union, List

def load_model_and_tokenizer_simplified(
    model_name: str,
    adapter_name: Union[str, None] = None,
    padding_side: str = 'right',
    template_name: str = 'llama2'
):
    '''
        Responsible for loading model and tokenizer
    ''' 

    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side=padding_side
    )

    # loading the model and applying the adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    if adapter_name:
        model = PeftModel.from_pretrained(model, adapter_name, is_trainable=False)

    # fixing the tokenizer
    _ = get_template_and_fix_tokenizer(tokenizer, template_name)

    return model, tokenizer


def create_essay_instruction(
    tokenizer: AutoTokenizer,
    title: str,
    template: str = 'default'
):
    '''
        Function for creating essay instruction based on title
    '''
    message_list = [
        {'role': Role.USER.value, 'content': title},
        {'role': Role.ASSISTANT.value, 'content': 'Dummy message'}
    ]
    prompt_ids, _ = templates[template].encode_oneturn(
        tokenizer,
        message_list,
        system=(
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        cutoff_len=4096
    )
    return prompt_ids

def create_feedback_instruction(
    tokenizer: AutoTokenizer,
    instruction: str,
    title: str,
    content: str,
    template: str = 'llama2'
):
    '''
        Function for creating feedback instruction based on title and feedback
    '''

    # checking whether title is present sequentially
    essay = ''
    if title:
        essay += title + '\n\n'
    essay += content
    content = '{instruction}\n{content}'.format(
        instruction='',
        content=essay
    )

    message_list = [
        {'role': Role.USER.value, 'content': content},
        {'role': Role.ASSISTANT.value, 'content': 'Dummy message'}
    ]
    prompt_ids, _ = templates[template].encode_oneturn(
        tokenizer,
        message_list,
        cutoff_len=4096,
        system=instruction
    )
    return prompt_ids

def create_student_applicator_instruction(
    tokenizer: AutoTokenizer,
    instruction: str,
    title: str,
    content: str,
    feedback_list: List[str],
    template: str = 'llama2'
):
    '''
        Function for creating student applicator instruction based on title
    '''

    # creating the essay string
    essay = ''
    if title:
        essay += title + '\n\n'
    essay += content

    # creating the feedback string
    feedback_string = ''
    for index, feedback in enumerate(feedback_list):
        feedback_string += 'Reviewer {}\n{}\n\n'.format(index + 1, feedback)

    # final content string
    content = '{instruction}\nEssay: {content}\n\nFeedback: {feedback}'.format(
        instruction='',
        content=essay,
        feedback=feedback_string
    )

    message_list = [
        {'role': Role.USER.value, 'content': content},
        {'role': Role.ASSISTANT.value, 'content': 'Dummy message'}
    ]
    prompt_ids, _ = templates[template].encode_oneturn(
        tokenizer,
        message_list,
        cutoff_len=4096,
        system=instruction
    )
    return prompt_ids

if __name__ == '__main__':

    # loading a model and tokenizer at a certain path
    model, tokenizer = load_model_and_tokenizer_simplified(
        '/home/inair/data/llama3_models/llama-3-8b-hf',
        '/home/inair/data/revision_saves/sft_mwrite_student_applicator_8b'
    )

    import pdb; pdb.set_trace()

    essay = "Greetings, Board of Supervisors Members.\nI'm writing to say my objections to Mr. Yee's suggested policy, which would outlaw automation. As an economics student and concerned citizen about to enter the workforce, I think that such a legislation would negatively impact San Francisco's labor market and manufacturing efficiency.\nFirst, it's important to comprehend how raising the minimum wage would affect the job market economically. Employment rates decline and unemployment will start to rise when the minimum wage is raised to a binding level, which is higher than the equilibrium wage rate. In other words this means that the minimum wage would be increased over the equilibrium making it harder for the business to hire as many workers as before. Because of this businesses will start to lower their demand for labor in response to rising labor expenses, which will lead them to look for alternate ways of production, such as automation, to maintain competitiveness in the market.\nThe proposed regulations will cause a ton of negative effects of raising the minimum wage in San Francisco, where there is a general fear of job loss due to automation already. Businesses would have greater production costs if automation technology was restricted. This might eventually result in poorer profitability and possibly even businesses closing. Furthermore, prohibiting automation stunts productivity growth and innovation, two key factors that propel economic development. However, since I am concerned about what my life and a lot of other young workers' lives are going to be like after automation I don\u2019t think the wage increase is necessary. If I were to be looking for a job right now I would not be able to find one because of the binding wage increase causing the supply of workers in the labor market to decrease. \nFurthermore, the ban on automation would mess up market dynamics and delay consumer happiness. By limiting the use of automation, businesses would be forced to rely more heavily on labor methods of production, leading to higher prices for goods and services. This would result in a reduction in consumer surplus, as consumers are forced to pay higher prices for products that could have been produced more efficiently through automation. However this will lead to a greater producer surplus and more deadweight loss. \nIn conclusion, while I understand the intention behind Mr. Yee's proposal to protect jobs in the face of automation, I believe that it is misguided and would ultimately harm both businesses and consumers in San Francisco. Instead of trying to stop innovation, the policymakers should focus on strategies to support workers in transitioning to new opportunities and encourage an environment favorable towards technological advancement.\nThank you for considering my perspective on this important issue.\n\n[]"

    feedback_list = [
        "Understanding 1: Based on your class discussion and course readings, identify any important concepts that are missing. Identify any unnecessary concepts in use.: One major concept that was referred to but was never actually stated was a price floor. A price floor is what sets a minimum price that must be paid. Another concept the writer could mention is an increase in inputs as the reason San Francisco companies would raise their prices. Input prices are an example of what causes producers to decrease their supply of their product or decrease their demand for labor.\nUnderstanding 2: How can the author connect concepts in a more useful manner? For example, using your knowledge from class, how could the author improve their explanation of interactions between the various markets affected by these policies?: The writer could refer to the raise in minimum wage being a binding price floor in the labor market. The writer could also talk about how a raise on minimum wage would cause an increase in inputs for these companies. This could then be used to explain why minimum wage would cause a decrease in demand for labor, an increase in product price, and/or a decrease in product supply.\nCritical Thinking 1: Based on your class discussion and course readings, how could the author improve their analysis of the minimum wage increase and the automation policy (ban or tax, depending)?: One thing I think the writer could improve was the part where they said \u201cIf I were to be looking for a job right now I would not be able to find one because of the binding wage increase causing the supply of workers in the labor market to decrease.\u201d The binding wage increase would not cause the supply of workers in the labor market to decrease because the supply will stay the same, but the number of unemployed workers would increase. A suggestion I\u2019d have for this part is to possibly switch this statement with one about how the increase in minimum wage would create a rise in unemployment and increase the chance of us (new workers to the market) being unemployed.\nCritical Thinking 2: How well does the author apply economic principles to justify his/her position? Suggest one (or two) additional ways the author could apply economics to their argument in order to make this letter more persuasive.: I think the author does well with applying economic principles to justify their position. Again, one thing that could be changed is the part when the writer says that a binding wage increase would cause the supply of workers in the labor market to decrease when what would actually happen is a decrease in the demand for workers in the labor market. The author could also say that this would be caused by an increase in input prices for the companies hiring these workers. Other than that, I think the writer did well using economic principles to justify their position.\nCritical Thinking 3: Are all outside sources properly cited?: The writer did not cite the San Francisco article. They did not use any other outside sources.\nResponse Alignment with Audience: The letter should be understandable to a person with a basic but not sophisticated understanding of economic principles. In this context, which parts were difficult to understand? Which parts were easy to understand?: I did not find any parts of the writer\u2019s paper to be difficult to understand. The author used basic economic principles to make their argument. In the second to last paragraph, the author does well with describing how a ban on automation would increase prices for consumers and change the amount of surplus in the market.",
        "Understanding 1: Based on your class discussion and course readings, identify any important concepts that are missing. Identify any unnecessary concepts in use.: I think that they touch upon most of the concepts that apply to this topic. He covered unemployment rates, supply and demand, and all the topics involved with efficiency (CS, PS, and deadweight loss).\nUnderstanding 2: How can the author connect concepts in a more useful manner? For example, using your knowledge from class, how could the author improve their explanation of interactions between the various markets affected by these policies?: The author does a good job of explaining the specific results within each market, touching on consumer surplus and producer surplus, and they mention the market interactions, however, I would have liked to see a more explicit explanation. For example, I want them to mention the curves within each market and how shifts/movements in one affect the other curves in the other markets.\nCritical Thinking 1: Based on your class discussion and course readings, how could the author improve their analysis of the minimum wage increase and the automation policy (ban or tax, depending)?: I think that the author did a great job, however, they could have created a more coherent chain of events, as I mentioned earlier, and shown more understanding of how automation and labor markets affect each other. They could achieve this by talking about how the supply and demand of labor markets affect consumer prices.\nCritical Thinking 2: How well does the author apply economic principles to justify his/her position? Suggest one (or two) additional ways the author could apply economics to their argument in order to make this letter more persuasive.: I think that one way the author could justify his/her position is by specifically showing some marginal analysis, and highlighting marginal costs incurred when hiring workers versus automation. This would show that the author has a better understanding of the situation at hand.\nCritical Thinking 3: Are all outside sources properly cited?: The sources are not properly cited. Most of this information is from the article or Dr. Dudley\u2019s lectures, and none is written anywhere in the letter.\nResponse Alignment with Audience: The letter should be understandable to a person with a basic but not sophisticated understanding of economic principles. In this context, which parts were difficult to understand? Which parts were easy to understand?: This letter is understandable to a person with a basic but not sophisticated understanding of economic principles. The author does a phenomenal job all around in breaking down concepts in an easy-to-understand manner, while still conveying the economic principles. For example, in the second paragraph, the author mentions how raising the minimum wage could create unemployment and decrease employment rates, and then further breaks down why this will happen, citing specifically that companies won\u2019t be able to hire at the same rate and how it will lower their demand for labor, causing them to seek alternative sources.",
        "Understanding 1: Based on your class discussion and course readings, identify any important concepts that are missing. Identify any unnecessary concepts in use.: This person has all of the correct ideas. They are extremely clear, and offer a persuasive and concise explanation of their ideas. The one thing they could add to strengthen their paper is to consider businesses relocating, further stalling the market.\nUnderstanding 2: How can the author connect concepts in a more useful manner? For example, using your knowledge from class, how could the author improve their explanation of interactions between the various markets affected by these policies?: The author does a great job connecting and building concepts. The one thing they could do to connect the paper more is to connect the end of the paper back to the beginning of the paper. Making the paper go full circle will give the reader a feeling of completion at the end.\nCritical Thinking 1: Based on your class discussion and course readings, how could the author improve their analysis of the minimum wage increase and the automation policy (ban or tax, depending)?: The author does what he needs to when explaining these topics. They could improve their explanation of minimum wage by describing the shift in supply and demand of jobs a little more clearly. Overall, they are extremely clear.\nCritical Thinking 2: How well does the author apply economic principles to justify his/her position? Suggest one (or two) additional ways the author could apply economics to their argument in order to make this letter more persuasive.: They use economic principles to support their ideas extremely well. There are little to no holes in their argument. The one thing they could add to further support their argument is an example of how a company would react to these taxes.\nCritical Thinking 3: Are all outside sources properly cited?: The outside source mentioned in the prompt is not cited.\nResponse Alignment with Audience: The letter should be understandable to a person with a basic but not sophisticated understanding of economic principles. In this context, which parts were difficult to understand? Which parts were easy to understand?: The whole paper was fairly easy to understand. They connected their ideas extremely well, and had lots of persuasive points made in their paper. There was one point made in the beginning, where the author tried to describe the shift in the supply and demand curve. This was the only part in the paper that was moderately difficult to understand."
    ]

    instruction = "You are asked to go the distance and fully revise your first draft given your peer review experience. Meaningful revision means changes at the sentence and paragraph level, and not simply changing word choice.\n# Revision guidelines\n- Reread the following situation and prompt:\n    ## THE SITUATION\n    In San Francisco, fear of job loss due to automation following a minimum wage increase has led to two policy proposals: a tax on automation (Ms. Kim\u2019s proposal) and a ban on automation (Mr. Yee\u2019s proposal). As an economics student and soon-to-be job market entrant, you find this issue relevant.\n\n    ## THE PROMPT\n    Write a letter to the San Francisco Board of Supervisors about the two policy proposals: a tax on automation (Ms. Kim) or a ban on automation (Mr. Yee). Your letter should:\n    - Explain the economic impact of a minimum wage increase on the labor market, automation market, and the market for goods produced using labor and automation.\n    - Choose one policy and argue against it using economic principles learned in class, focusing on positive economic analysis rather than moral or ethical reasoning.\n    - Assume the minimum wage increase has already occurred.\n    - Keep in mind the Board's general knowledge of economics, including Supply, Demand, Consumer Surplus, Producer Surplus, and Efficiency, but explain specific interactions.\n\n    Cite the provided article and any external references used. Your letter should be 400-500 words, professionally written, and signed as \u201cA Concerned Citizen\u201d for anonymity. Include a word count at the end.\n- Reread the essay rubrics\n   ## Essay Rubric\n\n    ### Understanding\n    - **Concepts & Accuracy**:\n    - **Missing elements**: Several central economic terms/definitions missing. (1-3 Points)\n    - **Meets expectations**: Almost all central economic concepts identified/correctly defined. (4 Points)\n    - **Exceeds expectations**: All central economic concepts identified and defined exceptionally. (5 Points)\n\n    - **Linking Concepts**:\n    - **Missing elements**: Several connections between concepts and markets missing/incorrect. (1-3 Points)\n    - **Meets expectations**: Mostly correct connections between concepts and markets. (4 Points)\n    - **Exceeds expectations**: Sophisticated connections between concepts and markets. (5 Points)\n\n    - **Conciseness**:\n    - **Missing elements**: Over 510 words, many irrelevant sentences/words. (1-3 Points)\n    - **Meets expectations**: Within 510 words, some wordiness. (4 Points)\n    - **Exceeds expectations**: Succinctly answers in less than 500 words. (5 Points)\n\n    ### Critical Thinking\n    - **Interpreting Sources**:\n    - **Missing elements**: Limited understanding of sources; no citations. (1-3 Points)\n    - **Meets expectations**: Interprets sources correctly, mostly quotes, attempts citations. (4 Points)\n    - **Exceeds expectations**: Accurately interprets sources, mostly summarizes, proper citations. (5 Points)\n\n    - **Analysis of Case Study**:\n    - **Missing elements**: Missing policy opposition or economic impact acknowledgment. (1-3 Points)\n    - **Meets expectations**: Effectively debunks proposed solution, partial market interaction analysis. (4 Points)\n    - **Exceeds expectations**: Insightful issues articulation, explores all market interactions, interweaves concepts. (5 Points)\n\n    ### Response Alignment With Audience\n    - **Missing elements**: Misaligned explanations/recommendations, reasonable economic analysis. (1-3 Points)\n    - **Meets expectations**: Generally aligns with audience needs, format mostly correct. (4 Points)\n    - **Exceeds expectations**: Perfect format, consistently meets audience needs. (5 Points)\n"

    prompt_ids = create_student_applicator_instruction(
        tokenizer,
        instruction,
        '',
        essay,
        feedback_list,
        'llama3'
    )

    import pdb; pdb.set_trace()
    

