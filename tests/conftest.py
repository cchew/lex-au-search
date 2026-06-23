import pytest
from lexausearch.models import Chunk

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

CORPUS_INDEX = {"Privacy Act 1988": "/akn/au/act/1988/119/eng@2026-01-01"}

PRIVACY_ACT_XML_V3 = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork>
          <FRBRthis value="/akn/au/act/1988/119/!main"/>
          <FRBRuri value="/akn/au/act/1988/119"/>
          <FRBRdate date="1988" name="Generation"/>
          <FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/>
          <FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/>
          <FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-6">
        <num>6</num>
        <heading>Notification</heading>
        <content><p>The entity must notify the individual.</p></content>
        <authorialNote placement="end" marker="*">
          <content><p>Note: See also section 3.</p></content>
        </authorialNote>
      </section>
      <section eId="sec-7">
        <num>7</num>
        <heading>Fees</heading>
        <content>
          <table>
            <tr><th>Fee type</th><th>Amount</th></tr>
            <tr><td>Application fee</td><td>100 penalty units</td></tr>
          </table>
        </content>
      </section>
    </body>
    <attachments>
      <attachment>
        <hcontainer name="schedule" eId="schedule-1">
          <heading>Australian Privacy Principles</heading>
          <hcontainer name="clause" eId="schedule-1__clause-1">
            <num>1</num>
            <heading>Open and transparent management</heading>
            <content><p>APP entities must manage personal information openly.</p></content>
          </hcontainer>
          <hcontainer name="clause" eId="schedule-1__clause-2">
            <num>2</num>
            <heading>Anonymity and pseudonymity</heading>
            <content><p>Individuals must have the option of not identifying themselves.</p></content>
          </hcontainer>
        </hcontainer>
      </attachment>
    </attachments>
  </act>
</akomaNtoso>"""

PRIVACY_ACT_XML = f"""\
<?xml version='1.0' encoding='UTF-8'?>
<akomaNtoso xmlns="{AKN_NS}">
  <act name="act">
    <meta>
      <identification source="#lex-au">
        <FRBRWork>
          <FRBRthis value="/akn/au/act/1988/119/!main"/>
          <FRBRuri value="/akn/au/act/1988/119"/>
          <FRBRdate date="1988" name="Generation"/>
          <FRBRauthor href="#parliament"/>
        </FRBRWork>
        <FRBRExpression>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01"/>
          <FRBRdate date="2026-01-01" name="Generation"/>
          <FRBRauthor href="#parliament"/>
          <FRBRlanguage language="eng"/>
        </FRBRExpression>
        <FRBRManifestation>
          <FRBRthis value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRuri value="/akn/au/act/1988/119/eng@2026-01-01/!main.akn"/>
          <FRBRdate date="2026-06-19" name="Generation"/>
          <FRBRauthor href="#lex-au"/>
        </FRBRManifestation>
      </identification>
    </meta>
    <body>
      <section eId="sec-3">
        <num>3</num>
        <heading>Interpretation</heading>
        <content>
          <p>In this Act personal information means information about an identified individual.</p>
        </content>
      </section>
      <section eId="sec-6">
        <num>6</num>
        <heading>Meaning of personal information</heading>
        <content>
          <p>Personal information means information or an opinion about an identified individual.</p>
          <p>The information need not be recorded in material form.</p>
        </content>
      </section>
    </body>
  </act>
</akomaNtoso>"""


@pytest.fixture(scope="module")
def privacy_chunks() -> list[Chunk]:
    return [
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-3",
            provision_num="3",
            provision_type="section",
            heading="Interpretation",
            text="3 Interpretation In this Act personal information means information about an identified individual.",
            refs=[],
        ),
        Chunk(
            act_name="Privacy Act 1988",
            frbr_uri="/akn/au/act/1988/119/eng@2026-01-01",
            eid="sec-6",
            provision_num="6",
            provision_type="section",
            heading="Meaning of personal information",
            text="6 Meaning of personal information Personal information means information or an opinion about an identified individual. The information need not be recorded in material form.",
            refs=[],
        ),
    ]
