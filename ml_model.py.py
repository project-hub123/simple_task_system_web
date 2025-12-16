from openai import OpenAI

class GPTModel:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate_task(self):
        prompt = (
            "Придумай учебную задачу по Python с функцией и примером её вызова."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip(), None
        except Exception as e:
            return None, str(e)

    def check_solution(self, task, solution):
        prompt = (
            "Вот задание:\n" + task +
            "\n\nВот код:\n```python\n" + solution + "\n```" +
            "\n\nСкажи, правильно ли решено и что можно улучшить."
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            return response.choices[0].message.content.strip(), None
        except Exception as e:
            return None, str(e)
