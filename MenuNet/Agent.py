from openai import OpenAI
import torch
import re


class Agent:
    def __init__(self, api_key, model, reasoning):
        self.model = model
        #MODELS: gpt-5-mini, gpt-5.2, gpt-4o
        self.reasoning_effort = reasoning
        self.client = OpenAI(api_key=api_key)

    def communicate(self, context, instructions="", reasoning_effort=None):
        prompt = context + "\n"
        message = ""
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort

        response = self.client.responses.create(
            model=self.model,
            #instructions=instructions,
            #temperature = self.temp,
            input=prompt,
            
            #These are NOT supported by 4o
            text = {"verbosity": "medium"},
            reasoning = {"effort": effort},
            #minimal, low, medium, high, etc
            service_tier="priority"
        )

        message = response.output_text
        #print(message)
        return message
    

class LLMBidder(Agent):
    def __init__(self, api_key, model, reasoning="minimal"):
        super().__init__(api_key=api_key, model=model, reasoning=reasoning)
        self.valuations = None
        self.bid = None

    def make_valuation(self, max_list):
        item1_max, item2_max = max_list
        valuation1 = item1_max * torch.rand((1,1,1)) 
        valuation2 = item2_max * torch.rand((1,1,1))
        self.valuations = torch.cat((valuation1, valuation2), dim=2).flatten()
        return self.valuations

    def make_bid(self, context=""):
        response = self.communicate(context)
        bid_text, reasoning = self._parse_bid_and_reasoning(response)
        self.bid = bid_text
        return bid_text, reasoning



    def make_bids(self, contexts):
        if isinstance(contexts, str):
            return [self.make_bid(contexts)]
        if not contexts:
            return []

        combined_prompt = (
            "You will be given multiple independent auction rounds. "
            "Return ONLY the bids and a brief reasoning for each round in the exact format:\n"
            "1R: <brief reasoning>\n1: (b1, b2)\n"
            "2R: <brief reasoning>\n2: (b1, b2)\n"
            "3R: <brief reasoning>\n3: (b1, b2)\n"
            "Do not add any extra text.\n"
        )
        for i, ctx in enumerate(contexts, start=1):
            combined_prompt += f"\nROUND {i}:\n{ctx}\n"

        response = self.communicate(combined_prompt)

        return self._parse_bids_and_reasonings(response, len(contexts))

    def _parse_bid_and_reasoning(self, text):
        num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        bid_match = re.search(rf"\(\s*({num})\s*,\s*({num})\s*\)", text)
        bid_text = f"({bid_match.group(1)}, {bid_match.group(2)})" if bid_match else ""

        reasoning = ""
        for line in text.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("reason:") or lower.startswith("reasoning:") or lower.startswith("why:"):
                reasoning = stripped.split(":", 1)[1].strip()
                break
        if not reasoning:
            inline = re.search(r"(?:reason(?:ing)?|why)\s*[:\-]\s*(.+)$", text, re.IGNORECASE)
            if inline:
                reasoning = inline.group(1).strip()

        return bid_text, reasoning

    def _parse_bids_and_reasonings(self, text, count):
        num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        results = [("", "")] * count
        bid_line_re = re.compile(
            rf"^\s*(\d+)\s*[:\-]\s*\(?\s*({num})\s*,\s*({num})\s*\)?"
        )
        reason_line_re = re.compile(r"^\s*(\d+)\s*R\s*[:\-]\s*(.+)$", re.IGNORECASE)
        inline_reason_re = re.compile(r"(?:reason(?:ing)?|why)\s*[:\-]\s*(.+)$", re.IGNORECASE)

        for line in text.splitlines():
            match = bid_line_re.match(line)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(results):
                    bid_text = f"({match.group(2)}, {match.group(3)})"
                    reasoning = results[idx][1]
                    inline_reason = inline_reason_re.search(line)
                    if inline_reason:
                        reasoning = inline_reason.group(1).strip()
                    results[idx] = (bid_text, reasoning)
                continue
            match = reason_line_re.match(line)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(results):
                    results[idx] = (results[idx][0], match.group(2).strip())

        if any(not r[0] for r in results):
            pair_re = re.compile(rf"\(\s*({num})\s*,\s*({num})\s*\)")
            fallback_pairs = [
                f"({m.group(1)}, {m.group(2)})" for m in pair_re.finditer(text)
            ]
            for i, val in enumerate(results):
                if not val[0] and i < len(fallback_pairs):
                    results[i] = (fallback_pairs[i], val[1])

        return results
        
    



    
